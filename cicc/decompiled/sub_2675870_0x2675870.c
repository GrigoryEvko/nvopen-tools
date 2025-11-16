// Function: sub_2675870
// Address: 0x2675870
//
__m128i *__fastcall sub_2675870(__m128i *a1, __int64 a2)
{
  __int64 v3; // rdx
  unsigned int v4; // r14d
  unsigned __int8 *v5; // rax
  int v6; // eax
  unsigned __int64 v7; // rax
  __int64 *(__fastcall *v8)(__int64 *); // rax
  _BYTE *v9; // rax
  _BYTE *v10; // rdx
  __m128i *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  __int64 v15; // rcx
  size_t v16; // [rsp+8h] [rbp-78h] BYREF
  _BYTE *v17; // [rsp+10h] [rbp-70h] BYREF
  size_t v18; // [rsp+18h] [rbp-68h]
  _QWORD v19[2]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE *v20[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v21[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)a2 + 72LL) & 3LL;
  if ( v3 == 3 )
  {
    v4 = 7;
  }
  else
  {
    v4 = 1;
    if ( v3 != 2 )
    {
      v4 = 0;
      v5 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a2 + 72LL) & 0xFFFFFFFFFFFFFFFCLL);
      if ( v5 )
      {
        v6 = *v5;
        if ( (_BYTE)v6 == 22 )
        {
          v4 = 6;
        }
        else if ( (_BYTE)v6 )
        {
          v4 = 1;
          if ( (unsigned __int8)v6 > 0x1Cu )
          {
            v7 = (unsigned int)(v6 - 34);
            if ( (unsigned __int8)v7 <= 0x33u )
            {
              v15 = 0x8000000000041LL;
              if ( _bittest64(&v15, v7) )
                v4 = 2 * ((_BYTE)v3 != 1) + 3;
            }
          }
        }
        else
        {
          v4 = 2 * ((_BYTE)v3 != 1) + 2;
        }
      }
    }
  }
  v20[0] = v21;
  sub_2240A50((__int64 *)v20, 1u, 45);
  sub_2554A60(v20[0], 1, v4);
  v8 = *(__int64 *(__fastcall **)(__int64 *))(**(_QWORD **)a2 + 72LL);
  if ( v8 == sub_2671190 )
  {
    v16 = 17;
    v17 = v19;
    v9 = (_BYTE *)sub_22409D0((__int64)&v17, &v16, 0);
    v17 = v9;
    v19[0] = v16;
    *(__m128i *)v9 = _mm_load_si128((const __m128i *)&xmmword_438FCC0);
    v10 = v17;
    v9[16] = 110;
    v18 = v16;
    v10[v16] = 0;
  }
  else
  {
    v8((__int64 *)&v17);
  }
  v11 = (__m128i *)sub_2241130((unsigned __int64 *)v20, 0, 0, v17, v18);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
  {
    a1[1] = _mm_loadu_si128(v11 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v11->m128i_i64[0];
    a1[1].m128i_i64[0] = v11[1].m128i_i64[0];
  }
  v12 = v11->m128i_i64[1];
  v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
  v13 = v17;
  v11->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v12;
  v11[1].m128i_i8[0] = 0;
  if ( v13 != v19 )
    j_j___libc_free_0((unsigned __int64)v13);
  if ( (_QWORD *)v20[0] != v21 )
    j_j___libc_free_0((unsigned __int64)v20[0]);
  return a1;
}
