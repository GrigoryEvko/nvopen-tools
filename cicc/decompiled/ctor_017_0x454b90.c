// Function: ctor_017
// Address: 0x454b90
//
int ctor_017()
{
  _QWORD v1[2]; // [rsp+0h] [rbp-10h] BYREF

  v1[0] = "Disable autoupgrade of debug info";
  v1[1] = 33;
  sub_A89F20(&unk_4F80C60, "disable-auto-upgrade-debug-info", v1);
  return __cxa_atexit(sub_984900, &unk_4F80C60, &qword_4A427C0);
}
