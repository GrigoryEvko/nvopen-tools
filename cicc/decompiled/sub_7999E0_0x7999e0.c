// Function: sub_7999E0
// Address: 0x7999e0
//
__int64 __fastcall sub_7999E0(__m128i *a1, __int64 a2, const __m128i *a3, _DWORD *a4)
{
  __int64 v6; // rax
  __int64 v8; // r12
  __int32 v9; // edi
  __int32 v10; // esi
  __int64 v11; // rcx
  __int64 v12; // r12
  unsigned int v13; // edx
  _DWORD *j; // rax
  __int64 result; // rax
  __int64 v16; // rcx
  __int32 v17; // edi
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 *i; // r12
  __int64 v21; // r13

  v6 = *(_QWORD *)(a2 + 8);
  v8 = *(_QWORD *)(a2 + 24);
  if ( v6 )
    sub_77A750((__int64)a1, *(_QWORD *)(v6 + 8));
  if ( v8 )
  {
    if ( *(_BYTE *)(v8 + 40) == 20 )
    {
      for ( i = *(__int64 **)(v8 + 72); i; i = (__int64 *)*i )
      {
        if ( *((_BYTE *)i + 8) == 7 )
          sub_77A750((__int64)a1, i[2]);
      }
    }
  }
  if ( a1[3].m128i_i64[0] && *a4 )
    *a4 = sub_799890((__int64)a1);
  v9 = a1[2].m128i_i32[2];
  v10 = a1[4].m128i_i32[0];
  v11 = a1[3].m128i_i64[1];
  v12 = a1[2].m128i_i64[0];
  v13 = v10 & a1[2].m128i_i32[2];
  for ( j = (_DWORD *)(v11 + 4LL * v13); v9 != *j; j = (_DWORD *)(v11 + 4LL * v13) )
    v13 = v10 & (v13 + 1);
  *j = 0;
  if ( *(_DWORD *)(v11 + 4LL * ((v13 + 1) & v10)) )
    sub_771390(a1[3].m128i_i64[1], a1[4].m128i_i32[0], v13);
  --a1[4].m128i_i32[1];
  a1[1] = _mm_loadu_si128(a3);
  a1[2] = _mm_loadu_si128(a3 + 1);
  result = a3[2].m128i_i64[0];
  a1[3].m128i_i64[0] = result;
  if ( v12 && v12 != a3[1].m128i_i64[0] )
  {
LABEL_14:
    v16 = *(unsigned int *)(v12 + 12);
    v17 = a1[4].m128i_i32[0];
    v18 = a1[3].m128i_i64[1];
    result = (unsigned int)v17 & *(_DWORD *)(v12 + 12);
    v19 = *(unsigned int *)(v18 + 4 * result);
    if ( (_DWORD)v16 )
    {
      while ( (_DWORD)v16 != (_DWORD)v19 )
      {
        if ( !(_DWORD)v19 )
        {
          v21 = *(_QWORD *)v12;
          result = sub_822B90(v12, *(unsigned int *)(v12 + 8), v19, v16);
          if ( v21 )
          {
            v12 = v21;
            goto LABEL_14;
          }
          v12 = 0;
          break;
        }
        result = v17 & (unsigned int)(result + 1);
        v19 = *(unsigned int *)(v18 + 4LL * (unsigned int)result);
      }
    }
    a1[2].m128i_i64[0] = v12;
  }
  return result;
}
