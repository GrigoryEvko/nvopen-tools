// Function: sub_86F7A0
// Address: 0x86f7a0
//
unsigned int *__fastcall sub_86F7A0(__int64 a1, unsigned int *a2)
{
  _BYTE *v2; // rax

  v2 = sub_86E480(0x15u, a2);
  *((_QWORD *)v2 + 9) = a1;
  return sub_86F5D0((__int64)v2);
}
