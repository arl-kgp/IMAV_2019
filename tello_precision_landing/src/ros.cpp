#include "helpers.h"
#include "ros.h"
#include "detect.h"
#include "ibvs.h"

Mat buffer_frame;
uint64_t fc = 0;
uint64_t cfc = -1;
bool cp = false;
zmq::context_t zcontext(1);
void imageCallback()
{
  //cout << "frame received\n";

    //  Socket to talk to server
    zmq::socket_t subscriber(zcontext, ZMQ_SUB);
    subscriber.connect("tcp://localhost:5556");
    subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    while (1)
    {
        zmq::message_t update;
        subscriber.recv(&update);
        image im;
        im.ParseFromArray((const void *)update.data(), update.size());
        Mat im2(Size(im.width(), im.height()), CV_8UC3, (void*)im.image_data().c_str());
        flip(im2, im2, 0);
        //flip(im2, im2, 1);
        while(cp)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        buffer_frame = im2;

        fc++;
    }
}

void get_quatbebo()
{
    zmq::socket_t subscriber(zcontext, ZMQ_SUB);
    subscriber.connect("tcp://localhost:5558");
    subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    while (1)
    {
        zmq::message_t update;
        subscriber.recv(&update);
        depth_m d;
        d.ParseFromArray((const void *)update.data(), update.size());
        depth = d.d();
        if(depth == 0) depth = 1;
    }
}

void rqt_plot(float s_norm)
{
   /* std_msgs::Float32 sf;
    sf.data = s_norm;
    if(s_norm <= 0.5)
    {
        cout << "Landinggggggg!";
        land.publish(std_msgs::Empty());
    }
    rqt.publish(sf);*/
}
zmq::socket_t vpub(zcontext, ZMQ_PUB);

void publish_vel_forward()
{
    vel v;
    v.set_vx(50);
    v.set_vy(0);
    v.set_vz(0);
    v.set_rz(0);

    cout << "Velocity: " << v.vx() << " " << v.vy() << " " << v.vz() << " " << v.rz() << "\n";
    zmq::message_t m;
    size_t size = v.ByteSizeLong(); 
    void *buffer = malloc(size);
    v.SerializeToArray(buffer, size);
    //vpub.send(buffer, size);
    free(buffer);
}

void publish_vel(MatrixXf veli)
{
 //   if(vel.norm() > 0.1)
 //       vel = vel/vel.norm()*0.1;
    //veli *= 0.1;
    vel v;
    v.set_vx(-veli(1,0));
    v.set_vy(-veli(0,0));
    v.set_vz(-veli(2,0));
    v.set_rz(-veli(3,0));

    cout << "Velocity: " << v.vx() << " " << v.vy() << " " << v.vz() << " " << v.rz() << "\n";
    zmq::message_t m;
    size_t size = v.ByteSizeLong(); 
    void *buffer = malloc(size);
    v.SerializeToArray(buffer, size);
    vpub.send(buffer, size);
    free(buffer);
}

Mat get_new_frame()
{
    while (fc == cfc || fc == 0)
    {
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
    //cout << 1;
    //zcontext = zmq::context_t(1);
    std::thread([](){ imageCallback(); }).detach();
    std::thread([](){ get_quatbebo(); }).detach();
    //vpub = zmq::socket_t(zcontext, ZMQ_PUB);
    vpub.connect("tcp://localhost:5557");
    cout << "ZMQ Initiated!\n";

    int counter = 0;
    reference.push_back(Point2d(376, 355));
    reference.push_back(Point2d(469, 353));
    reference.push_back(Point2d(473, 471));
    reference.push_back(Point2d(378, 472));
    reference.push_back(Point2d(399, 354));
    reference.push_back(Point2d(444, 353));
    reference.push_back(Point2d(448, 471));
    reference.push_back(Point2d(402, 473));
    reference.push_back(Point2d(436, 393));
    reference.push_back(Point2d(438, 433));
    reference.push_back(Point2d(411, 434));
    reference.push_back(Point2d(409, 393));
    cout << "start\n";
    startDetect(true);

    //ros::spin();
    return 0;
}
