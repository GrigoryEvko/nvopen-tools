// Function: sub_ED1420
// Address: 0xed1420
//
__int64 *__fastcall sub_ED1420(__int64 *a1, _BYTE *a2, __int64 a3, int a4, _BYTE *a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __m128i *v14; // rax
  __int64 v15; // rcx
  _OWORD *v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // [rsp+0h] [rbp-70h] BYREF
  __int64 v19; // [rsp+8h] [rbp-68h]
  _QWORD v20[2]; // [rsp+10h] [rbp-60h] BYREF
  _OWORD *v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  _OWORD v23[4]; // [rsp+30h] [rbp-40h] BYREF

  v7 = (__int64)&a2[a3];
  if ( *a2 == 1 )
  {
    v7 = (__int64)a2;
    if ( a3 )
    {
      v7 = (__int64)&a2[a3];
      ++a2;
    }
  }
  *a1 = (__int64)(a1 + 2);
  sub_ED0450(a1, a2, v7);
  if ( (unsigned int)(a4 - 7) <= 1 )
  {
    if ( a6 )
    {
      if ( a5 )
      {
        v18 = v20;
        sub_ED0450((__int64 *)&v18, a5, (__int64)&a5[a6]);
        if ( v19 == 0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
      }
      else
      {
        v19 = 0;
        v18 = v20;
        LOBYTE(v20[0]) = 0;
      }
      v14 = (__m128i *)sub_2241490(&v18, ":", 1, v11);
      v21 = v23;
      if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
      {
        v23[0] = _mm_loadu_si128(v14 + 1);
      }
      else
      {
        v21 = (_OWORD *)v14->m128i_i64[0];
        *(_QWORD *)&v23[0] = v14[1].m128i_i64[0];
      }
      v15 = v14->m128i_i64[1];
      v14[1].m128i_i8[0] = 0;
      v22 = v15;
      v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
      v16 = v21;
      v14->m128i_i64[1] = 0;
      v17 = sub_2241130(a1, 0, 0, v16, v22);
      sub_2240AE0(a1, v17);
      if ( v21 != v23 )
        j_j___libc_free_0(v21, *(_QWORD *)&v23[0] + 1LL);
      if ( v18 != v20 )
        j_j___libc_free_0(v18, v20[0] + 1LL);
    }
    else
    {
      v12 = sub_2241130(a1, 0, 0, "<unknown>:", 10);
      sub_2240AE0(a1, v12);
    }
  }
  return a1;
}
