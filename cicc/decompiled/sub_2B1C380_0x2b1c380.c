// Function: sub_2B1C380
// Address: 0x2b1c380
//
void __fastcall sub_2B1C380(
        unsigned int *src,
        unsigned int *a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7,
        __int64 a8)
{
  __int64 v10; // rdx
  unsigned int *v11; // r14
  unsigned int *v12; // r15
  unsigned int *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int128 v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+20h] [rbp-40h]

  v10 = (char *)a2 - (char *)src;
  v11 = (unsigned int *)&a3[(char *)a2 - (char *)src];
  v20 = (char *)a2 - (char *)src;
  v21 = a2 - src;
  v22 = (__int128)_mm_loadu_si128(&a7);
  v23 = a8;
  if ( (char *)a2 - (char *)src <= 24 )
  {
    sub_2B1C2A0(src, a2, v10, a4, a5, a6, *(_OWORD *)&a7, a8);
  }
  else
  {
    v12 = src;
    do
    {
      v13 = v12;
      v12 += 7;
      sub_2B1C2A0(v13, v12, v10, a4, a5, a6, v22, v23);
    }
    while ( (char *)a2 - (char *)v12 > 24 );
    sub_2B1C2A0(v12, a2, v10, a4, a5, a6, v22, v23);
    if ( v20 > 28 )
    {
      v16 = 7;
      do
      {
        sub_2B1C0F0(src, a2, a3, v16, v14, v15, (__int64 *)a7.m128i_i64[0]);
        v17 = 2 * v16;
        v16 *= 4;
        sub_2B1C0F0((unsigned int *)a3, v11, (char *)src, v17, v18, v19, (__int64 *)a7.m128i_i64[0]);
      }
      while ( v21 > v16 );
    }
  }
}
