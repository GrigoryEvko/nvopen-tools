// Function: sub_37DD570
// Address: 0x37dd570
//
void __fastcall sub_37DD570(char *a1, __m128i *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __m128i *v8; // rbx
  __int64 i; // r15
  __int64 v10; // r13
  __int32 v11; // eax
  __m128i v12; // xmm0
  int v13; // edx
  __m128i v14; // [rsp+20h] [rbp-50h]

  v6 = (char *)a2 - a1;
  v8 = a2;
  if ( (char *)a2 - a1 > 20 )
  {
    for ( i = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v6 >> 2) - 2) / 2; ; --i )
    {
      v14 = _mm_loadu_si128((const __m128i *)&a1[20 * i]);
      sub_37B6400(
        (__int64)a1,
        i,
        0xCCCCCCCCCCCCCCCDLL * (v6 >> 2),
        a4,
        a5,
        a6,
        v14.m128i_i64[0],
        v14.m128i_i64[1],
        *(_DWORD *)&a1[20 * i + 16]);
      if ( !i )
        break;
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    v10 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 2);
    do
    {
      v11 = *(_DWORD *)a1;
      if ( v8->m128i_i32[0] < *(_DWORD *)a1
        || v8->m128i_i32[0] == v11 && (a4 = *((unsigned int *)a1 + 1), v8->m128i_i32[1] < (unsigned int)a4) )
      {
        v12 = _mm_loadu_si128(v8);
        v8->m128i_i32[0] = v11;
        v13 = v8[1].m128i_i32[0];
        v8->m128i_i32[1] = *((_DWORD *)a1 + 1);
        v8->m128i_i32[2] = *((_DWORD *)a1 + 2);
        v8->m128i_i32[3] = *((_DWORD *)a1 + 3);
        v8[1].m128i_i32[0] = *((_DWORD *)a1 + 4);
        sub_37B6400((__int64)a1, 0, v10, a4, a5, a6, v12.m128i_i64[0], v12.m128i_i64[1], v13);
      }
      v8 = (__m128i *)((char *)v8 + 20);
    }
    while ( a3 > (unsigned __int64)v8 );
  }
}
