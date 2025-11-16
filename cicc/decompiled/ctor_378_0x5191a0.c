// Function: ctor_378
// Address: 0x5191a0
//
int ctor_378()
{
  _QWORD v1[2]; // [rsp+0h] [rbp-10h] BYREF

  v1[0] = "Abort when the max iterations for devirtualization CGSCC repeat pass is reached";
  v1[1] = 79;
  sub_227B4B0(&unk_4FDADE0, "abort-on-max-devirt-iterations-reached", v1);
  return __cxa_atexit(sub_984900, &unk_4FDADE0, &qword_4A427C0);
}
