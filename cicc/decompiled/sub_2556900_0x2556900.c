// Function: sub_2556900
// Address: 0x2556900
//
__m128i *__fastcall sub_2556900(__m128i *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r14
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  int v6; // eax
  __m128i *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v10[2]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v11; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v12[2]; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v13; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v14; // [rsp+50h] [rbp-90h] BYREF
  __int64 v15; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v16; // [rsp+70h] [rbp-70h] BYREF
  int v17; // [rsp+78h] [rbp-68h]
  _QWORD v18[2]; // [rsp+80h] [rbp-60h] BYREF
  __m128i v19; // [rsp+90h] [rbp-50h] BYREF
  __int64 v20; // [rsp+A0h] [rbp-40h] BYREF

  v2 = a2;
  v3 = *(unsigned int *)(a2 + 160);
  if ( v3 <= 9 )
  {
    a2 = 1;
  }
  else if ( v3 <= 0x63 )
  {
    a2 = 2;
  }
  else if ( v3 <= 0x3E7 )
  {
    a2 = 3;
  }
  else if ( v3 <= 0x270F )
  {
    a2 = 4;
  }
  else
  {
    v4 = *(unsigned int *)(a2 + 160);
    LODWORD(a2) = 1;
    while ( 1 )
    {
      v5 = v4;
      v6 = a2;
      a2 = (unsigned int)(a2 + 4);
      v4 /= 0x2710u;
      if ( v5 <= 0x1869F )
        break;
      if ( v5 <= 0xF423F )
      {
        a2 = (unsigned int)(v6 + 5);
        break;
      }
      if ( v5 <= (unsigned __int64)&loc_98967F )
      {
        a2 = (unsigned int)(v6 + 6);
        break;
      }
      if ( v5 <= 0x5F5E0FF )
      {
        a2 = (unsigned int)(v6 + 7);
        break;
      }
    }
  }
  v16 = v18;
  sub_2240A50((__int64 *)&v16, a2, 0);
  sub_1249540(v16, v17, v3);
  sub_2509010(v10, *(unsigned __int8 *)(v2 + 168));
  v7 = (__m128i *)sub_2241130((unsigned __int64 *)v10, 0, 0, "CallEdges[", 0xAu);
  v12[0] = (unsigned __int64)&v13;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    v13 = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    v12[0] = v7->m128i_i64[0];
    v13.m128i_i64[0] = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_u64[1];
  v7[1].m128i_i8[0] = 0;
  v12[1] = v8;
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v7->m128i_i64[1] = 0;
  sub_94F930(&v14, (__int64)v12, ",");
  sub_8FD5D0(&v19, (__int64)&v14, &v16);
  sub_94F930(a1, (__int64)&v19, "]");
  if ( (__int64 *)v19.m128i_i64[0] != &v20 )
    j_j___libc_free_0(v19.m128i_u64[0]);
  if ( (__int64 *)v14.m128i_i64[0] != &v15 )
    j_j___libc_free_0(v14.m128i_u64[0]);
  if ( (__m128i *)v12[0] != &v13 )
    j_j___libc_free_0(v12[0]);
  if ( (__int64 *)v10[0] != &v11 )
    j_j___libc_free_0(v10[0]);
  if ( v16 != (_BYTE *)v18 )
    j_j___libc_free_0((unsigned __int64)v16);
  return a1;
}
