// Function: sub_2509250
// Address: 0x2509250
//
__m128i *__fastcall sub_2509250(__m128i *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // esi
  unsigned __int8 *v7; // rax
  int v8; // eax
  unsigned __int64 v9; // rax
  __m128i *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // rdi
  __int64 v14; // rcx
  _BYTE *v15[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF
  __int64 v17[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v18; // [rsp+30h] [rbp-30h] BYREF

  v4 = *(_QWORD *)(*(_QWORD *)a2 + 72LL);
  v5 = v4 & 3;
  if ( v5 == 3 )
  {
    v6 = 7;
  }
  else
  {
    v6 = 1;
    if ( v5 != 2 )
    {
      v6 = 0;
      v7 = (unsigned __int8 *)(v4 & 0xFFFFFFFFFFFFFFFCLL);
      if ( v7 )
      {
        v8 = *v7;
        if ( (_BYTE)v8 == 22 )
        {
          v6 = 6;
        }
        else if ( (_BYTE)v8 )
        {
          v6 = 1;
          if ( (unsigned __int8)v8 > 0x1Cu )
          {
            v9 = (unsigned int)(v8 - 34);
            if ( (unsigned __int8)v9 <= 0x33u )
            {
              v14 = 0x8000000000041LL;
              if ( _bittest64(&v14, v9) )
                v6 = 2 * ((_BYTE)v5 != 1) + 3;
            }
          }
        }
        else
        {
          v6 = 2 * ((_BYTE)v5 != 1) + 2;
        }
      }
    }
  }
  sub_2509010(v17, v6);
  (*(void (__fastcall **)(_BYTE **))(**(_QWORD **)a2 + 72LL))(v15);
  v10 = (__m128i *)sub_2241130((unsigned __int64 *)v17, 0, 0, v15[0], (size_t)v15[1]);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
  {
    a1[1] = _mm_loadu_si128(v10 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v10->m128i_i64[0];
    a1[1].m128i_i64[0] = v10[1].m128i_i64[0];
  }
  v11 = v10->m128i_i64[1];
  v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
  v12 = (__int64 *)v15[0];
  v10->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v11;
  v10[1].m128i_i8[0] = 0;
  if ( v12 != &v16 )
    j_j___libc_free_0((unsigned __int64)v12);
  if ( (__int64 *)v17[0] != &v18 )
    j_j___libc_free_0(v17[0]);
  return a1;
}
