// Function: sub_26E2F60
// Address: 0x26e2f60
//
void __fastcall sub_26E2F60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 *a5, char a6, char a7)
{
  const __m128i *v11; // rdi
  _QWORD *v12; // rbx
  unsigned __int64 v13; // rdi
  const __m128i *v14; // [rsp+10h] [rbp-B0h] BYREF
  const __m128i *v15; // [rsp+18h] [rbp-A8h]
  __int64 v16; // [rsp+20h] [rbp-A0h]
  const __m128i *v17; // [rsp+30h] [rbp-90h] BYREF
  const __m128i *v18; // [rsp+38h] [rbp-88h]
  __int64 v19; // [rsp+40h] [rbp-80h]
  __m128i s; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v21; // [rsp+60h] [rbp-60h]
  __int64 v22; // [rsp+68h] [rbp-58h]
  char v23; // [rsp+80h] [rbp-40h] BYREF

  if ( a6 || a7 )
  {
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    sub_26E1700(a1, a3, a4, (unsigned __int64 *)&v17, (unsigned __int64 *)&v14);
    v11 = v17;
    if ( v17 == v18
      || v14 == v15
      || (unsigned int)qword_4FF8568 < 0xAAAAAAAAAAAAAAABLL * (((char *)v18 - (char *)v17) >> 3)
      || (unsigned int)qword_4FF8568 < 0xAAAAAAAAAAAAAAABLL * (((char *)v15 - (char *)v14) >> 3) )
    {
      if ( !v17 )
        goto LABEL_10;
    }
    else
    {
      sub_26E28C0(&s, a1, &v17, &v14, a7);
      if ( a6 )
        sub_26E2D30(a1, &s, a3, a5);
      v12 = v21;
      while ( v12 )
      {
        v13 = (unsigned __int64)v12;
        v12 = (_QWORD *)*v12;
        j_j___libc_free_0(v13);
      }
      memset((void *)s.m128i_i64[0], 0, 8 * s.m128i_i64[1]);
      v22 = 0;
      v21 = 0;
      if ( (char *)s.m128i_i64[0] != &v23 )
        j_j___libc_free_0(s.m128i_u64[0]);
      v11 = v17;
      if ( !v17 )
        goto LABEL_10;
    }
    j_j___libc_free_0((unsigned __int64)v11);
LABEL_10:
    if ( v14 )
      j_j___libc_free_0((unsigned __int64)v14);
  }
}
