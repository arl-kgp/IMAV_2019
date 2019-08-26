#include "helpers.h"
#include "ros.h"
#include "detect.h"
#include "ibvs.h"

Mat buffer_frame;
uint64_t fc = 0;
uint64_t cfc = -1;
bool cp = false;
ros::Publisher motion;
ros::Publisher rqt;
ros::Publisher land;
void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cout << "frame received\n";
    auto i = cv_bridge::toCvCopy(msg);
    while (cp)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    buffer_frame = (*i).image;
    fc++;
}

void get_quatbebo(const nav_msgs::Odometry::ConstPtr &msg)
{

    depth = msg->pose.pose.position.z;
    if(depth == 0) depth = 1;
}

void rqt_plot(float s_norm)
{
    std_msgs::Float32 sf;
    sf.data = s_norm;
    if(s_norm <= 0.5)
    {
        cout << "Landinggggggg!";
        land.publish(std_msgs::Empty());
    }
  //  rqt.publish(sf);
}

void publish_vel(MatrixXf vel)
{
 //   if(vel.norm() > 0.1)
 //       vel = vel/vel.norm()*0.1;
    vel *= 0.1;
    geometry_msgs::Twist rot;
    geometry_msgs::Vector3 m;
    geometry_msgs::Vector3 s;
    s.x=s.z=s.y=0;
    float cap = 0.1 * depth*depth;
    m.x = -vel(1,0);
    m.y = -vel(0,0);
    m.z = -vel(2,0);
    s.z = -vel(3,0);
    if(abs(m.x) > cap)  m.x = abs(m.x)/m.x*cap;
    if(abs(m.y) > cap)  m.y = abs(m.y)/m.y*cap;
    if(abs(m.z) > cap)  m.z = abs(m.z)/m.z*cap;
    if(abs(s.z) > cap)  s.z = abs(s.z)/s.z*cap;
    cout << "Velocity: " << m.x << " " << m.y << " " << m.z << " " << s.z << "\n";

    rot.linear = m;
    rot.angular = s;
    //motion.publish(rot);
}

Mat get_new_frame()
{
    while (fc == cfc || fc == 0)
    {
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cp = true;
    cfc = fc;
    Mat rv;
    buffer_frame.copyTo(rv);
    cp = false;
    return rv;
}

int main(int argc, char **argv)
{
    //VideoCapture Video(1);
    int counter = 0;
    cout << "start\n";
    ros::init(argc, argv, "precision_landing");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ros::Subscriber sub2 = nh.subscribe("/bebop/odom", 1, get_quatbebo);
    image_transport::Subscriber sub = it.subscribe("/bebop/image_raw", 1, imageCallback);
    motion = nh.advertise<geometry_msgs::Twist>("/bebop/cmd_vel", 1);
    rqt = nh.advertise<std_msgs::Float32>("plot", 10);
    land = nh.advertise<std_msgs::Empty>("/bebop/land", 10);
    startDetect(false);

    ros::spin();
    return 0;
}
