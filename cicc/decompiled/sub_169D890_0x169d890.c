// Function: sub_169D890
// Address: 0x169d890
//
float __fastcall sub_169D890(__int64 *a1)
{
  float v2; // [rsp+Ch] [rbp-14h]
  float *v3; // [rsp+10h] [rbp-10h] BYREF
  unsigned int v4; // [rsp+18h] [rbp-8h]

  sub_169D7E0((__int64)&v3, a1);
  if ( v4 <= 0x40 )
    return *(float *)&v3;
  v2 = *v3;
  j_j___libc_free_0_0(v3);
  return v2;
}
