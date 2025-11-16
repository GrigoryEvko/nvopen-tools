// Function: ctor_087
// Address: 0x4a08f0
//
int ctor_087()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  int *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "How wide an instruction window to bypass looking for another guard";
  v3[1] = 66;
  v1 = 3;
  v2 = &v1;
  sub_10E1FD0(&unk_4F90120, "instcombine-guard-widening-window", &v2, v3);
  return __cxa_atexit(sub_984970, &unk_4F90120, &qword_4A427C0);
}
