// Function: sub_266B4A0
// Address: 0x266b4a0
//
void __fastcall sub_266B4A0(__int64 a1)
{
  char *v1; // r13
  __m128i *v2; // r12
  __int64 v3[2]; // [rsp+0h] [rbp-30h] BYREF
  char *v4; // [rsp+10h] [rbp-20h]

  v1 = *(char **)(a1 + 8);
  v2 = *(__m128i **)a1;
  sub_26695F0(v3, v2, (v1 - (char *)v2) >> 4);
  if ( v4 )
    sub_266B3D0(v2->m128i_i8, v1, v4, v3[1]);
  else
    sub_2665410(v2->m128i_i8, v1);
  j_j___libc_free_0((unsigned __int64)v4);
}
