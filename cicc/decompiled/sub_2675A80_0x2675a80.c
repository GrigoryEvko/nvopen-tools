// Function: sub_2675A80
// Address: 0x2675a80
//
__m128i *__fastcall sub_2675A80(__m128i *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  int v6; // eax
  __m128i *v7; // rax
  __int64 v8; // rcx
  __m128i *v9; // rax
  __int64 v10; // rcx
  _OWORD *v11; // rdi
  _BYTE *v13; // [rsp+0h] [rbp-60h] BYREF
  int v14; // [rsp+8h] [rbp-58h]
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  _OWORD *v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]
  _OWORD v18[3]; // [rsp+30h] [rbp-30h] BYREF

  v3 = *(unsigned int *)(a2 + 144);
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
    v4 = *(unsigned int *)(a2 + 144);
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
  v13 = v15;
  sub_2240A50((__int64 *)&v13, a2, 0);
  sub_1249540(v13, v14, v3);
  v7 = (__m128i *)sub_2241130((unsigned __int64 *)&v13, 0, 0, "[AAHeapToShared] ", 0x11u);
  v16 = v18;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    v18[0] = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    v16 = (_OWORD *)v7->m128i_i64[0];
    *(_QWORD *)&v18[0] = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_i64[1];
  v7[1].m128i_i8[0] = 0;
  v17 = v8;
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v7->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v17) <= 0x16 )
    sub_4262D8((__int64)"basic_string::append");
  v9 = (__m128i *)sub_2241490((unsigned __int64 *)&v16, " malloc calls eligible.", 0x17u);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
  {
    a1[1] = _mm_loadu_si128(v9 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v9->m128i_i64[0];
    a1[1].m128i_i64[0] = v9[1].m128i_i64[0];
  }
  v10 = v9->m128i_i64[1];
  v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
  v11 = v16;
  v9->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v10;
  v9[1].m128i_i8[0] = 0;
  if ( v11 != v18 )
    j_j___libc_free_0((unsigned __int64)v11);
  if ( v13 != (_BYTE *)v15 )
    j_j___libc_free_0((unsigned __int64)v13);
  return a1;
}
