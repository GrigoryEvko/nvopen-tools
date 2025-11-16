// Function: sub_1387510
// Address: 0x1387510
//
char __fastcall sub_1387510(const __m128i *a1, __m128i *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __m128i *v8; // rbx
  __int64 v9; // r14
  __int64 i; // r15
  __m128i v11; // xmm1
  unsigned int v12; // edx
  unsigned int v13; // esi
  unsigned __int32 v14; // r10d
  unsigned __int32 v15; // r11d
  __int64 v17; // [rsp+8h] [rbp-58h]
  __m128i v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+20h] [rbp-40h]

  v6 = (char *)a2 - (char *)a1;
  v8 = a2;
  v17 = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 > 24 )
  {
    v9 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
    for ( i = (v9 - 2) / 2; ; --i )
    {
      v18 = _mm_loadu_si128((const __m128i *)((char *)a1 + 24 * i));
      LOBYTE(v6) = (unsigned __int8)sub_13821F0(
                                      (__int64)a1,
                                      i,
                                      v9,
                                      a4,
                                      a5,
                                      a6,
                                      v18.m128i_i64[0],
                                      v18.m128i_i64[1],
                                      a1[1].m128i_i64[3 * i]);
      if ( !i )
        break;
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    while ( 1 )
    {
      v12 = v8->m128i_i32[0];
      LODWORD(v6) = a1->m128i_i32[0];
      if ( v8->m128i_i32[0] < (unsigned __int32)a1->m128i_i32[0] )
        goto LABEL_7;
      v13 = v8->m128i_u32[1];
      a4 = a1->m128i_u32[1];
      if ( v13 < (unsigned int)a4 && v12 == (_DWORD)v6 )
        goto LABEL_7;
      a5 = v8->m128i_u32[2];
      v14 = v8->m128i_u32[3];
      a6 = a1->m128i_u32[2];
      v15 = a1->m128i_u32[3];
      if ( v12 > (unsigned int)v6 || v13 > (unsigned int)a4 && v12 == (_DWORD)v6 )
        goto LABEL_8;
      if ( (unsigned int)a6 > (unsigned int)a5 )
        goto LABEL_7;
      LOBYTE(v6) = (_DWORD)a6 == (_DWORD)a5;
      if ( v15 > v14 && (_DWORD)a6 == (_DWORD)a5 )
        goto LABEL_7;
      if ( (unsigned int)a6 < (unsigned int)a5 || v15 < v14 && (_DWORD)a6 == (_DWORD)a5 )
        goto LABEL_8;
      if ( a1[1].m128i_i64[0] > v8[1].m128i_i64[0] )
      {
LABEL_7:
        v11 = _mm_loadu_si128(v8);
        v19 = v8[1].m128i_i64[0];
        *v8 = _mm_loadu_si128(a1);
        v8[1].m128i_i64[0] = a1[1].m128i_i64[0];
        LOBYTE(v6) = (unsigned __int8)sub_13821F0(
                                        (__int64)a1,
                                        0,
                                        0xAAAAAAAAAAAAAAABLL * (v17 >> 3),
                                        a4,
                                        a5,
                                        a6,
                                        v11.m128i_i64[0],
                                        v11.m128i_i64[1],
                                        v19);
LABEL_8:
        v8 = (__m128i *)((char *)v8 + 24);
        if ( a3 <= (unsigned __int64)v8 )
          return v6;
      }
      else
      {
        v8 = (__m128i *)((char *)v8 + 24);
        if ( a3 <= (unsigned __int64)v8 )
          return v6;
      }
    }
  }
  return v6;
}
