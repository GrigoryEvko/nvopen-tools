// Function: sub_2675510
// Address: 0x2675510
//
__m128i *__fastcall sub_2675510(__m128i *a1, __int64 a2)
{
  __int64 v3; // rdx
  unsigned int v4; // r13d
  unsigned __int8 *v5; // rax
  int v6; // eax
  unsigned __int64 v7; // rax
  __m128i *v8; // rax
  __int64 v9; // rcx
  _QWORD *v10; // rdi
  __int64 v12; // rcx
  _BYTE *v13[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v15[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v16[6]; // [rsp+30h] [rbp-30h] BYREF

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
              v12 = 0x8000000000041LL;
              if ( _bittest64(&v12, v7) )
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
  v13[0] = v14;
  sub_2240A50((__int64 *)v13, 1u, 45);
  sub_2554A60(v13[0], 1, v4);
  v15[0] = v16;
  sub_266E6F0((__int64 *)v15, "AAICVTracker", (__int64)"");
  v8 = (__m128i *)sub_2241130((unsigned __int64 *)v13, 0, 0, v15[0], (size_t)v15[1]);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    a1[1] = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v8->m128i_i64[0];
    a1[1].m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v9 = v8->m128i_i64[1];
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v10 = v15[0];
  v8->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v9;
  v8[1].m128i_i8[0] = 0;
  if ( v10 != v16 )
    j_j___libc_free_0((unsigned __int64)v10);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0((unsigned __int64)v13[0]);
  return a1;
}
