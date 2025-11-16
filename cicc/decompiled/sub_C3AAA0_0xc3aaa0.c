// Function: sub_C3AAA0
// Address: 0xc3aaa0
//
float __fastcall sub_C3AAA0(__int64 *a1)
{
  float v2; // [rsp+Ch] [rbp-14h]
  float *v3; // [rsp+10h] [rbp-10h] BYREF
  unsigned int v4; // [rsp+18h] [rbp-8h]

  sub_C3A850((__int64)&v3, a1);
  if ( v4 <= 0x40 )
    return *(float *)&v3;
  v2 = *v3;
  j_j___libc_free_0_0(v3);
  return v2;
}
