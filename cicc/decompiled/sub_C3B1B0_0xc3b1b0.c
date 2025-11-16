// Function: sub_C3B1B0
// Address: 0xc3b1b0
//
unsigned int *__fastcall sub_C3B1B0(__int64 a1, double a2)
{
  unsigned int *result; // rax
  double v3; // [rsp+0h] [rbp-10h] BYREF
  unsigned int v4; // [rsp+8h] [rbp-8h]

  v4 = 64;
  v3 = a2;
  result = sub_C3AF00(a1, dword_3F657A0, (__int64 *)&v3);
  if ( v4 > 0x40 && v3 != 0.0 )
    return (unsigned int *)j_j___libc_free_0_0(*(_QWORD *)&v3);
  return result;
}
