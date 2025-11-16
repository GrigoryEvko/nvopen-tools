// Function: ctor_618
// Address: 0x58b2a0
//
int ctor_618()
{
  int v1; // [rsp+0h] [rbp-20h] BYREF
  int v2; // [rsp+4h] [rbp-1Ch] BYREF
  int *v3; // [rsp+8h] [rbp-18h] BYREF

  v3 = &v1;
  v1 = 4;
  v2 = 2;
  sub_30A4990(&unk_502E0E0, "max-devirt-iterations", &v2, &v3);
  return __cxa_atexit(sub_984970, &unk_502E0E0, &qword_4A427C0);
}
