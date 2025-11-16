// Function: sub_2831360
// Address: 0x2831360
//
__int64 __fastcall sub_2831360(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rcx
  __int64 result; // rax
  __int64 v9; // rdx
  char v10; // cl
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rax
  __m128i *v14; // rsi
  __m128i *v15; // rsi
  __int64 v16; // rdx
  __m128i v17; // [rsp+0h] [rbp-30h] BYREF

  v7 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    result = *(_QWORD *)(a1 - 8);
    v9 = result + v7;
  }
  else
  {
    v9 = a1;
    result = a1 - v7;
  }
  if ( v9 != result )
  {
    v10 = 0;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)result == a2 )
        {
          if ( a2 )
          {
            v11 = *(_QWORD *)(result + 8);
            **(_QWORD **)(result + 16) = v11;
            if ( v11 )
              *(_QWORD *)(v11 + 16) = *(_QWORD *)(result + 16);
          }
          *(_QWORD *)result = a3;
          v10 = 1;
          if ( a3 )
            break;
        }
        result += 32;
        if ( v9 == result )
          goto LABEL_14;
      }
      v12 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)(result + 8) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = result + 8;
      *(_QWORD *)(result + 16) = a3 + 16;
      v10 = 1;
      *(_QWORD *)(a3 + 16) = result;
      result += 32;
    }
    while ( v9 != result );
LABEL_14:
    if ( v10 )
    {
      v13 = *(_QWORD *)(a1 + 40);
      v17.m128i_i64[1] = a3 & 0xFFFFFFFFFFFFFFFBLL;
      v14 = *(__m128i **)(a4 + 8);
      v17.m128i_i64[0] = v13;
      result = *(_QWORD *)(a4 + 16);
      if ( v14 == (__m128i *)result )
      {
        sub_F38BA0((const __m128i **)a4, v14, &v17);
        result = *(_QWORD *)(a1 + 40);
        v17.m128i_i64[1] = a2 | 4;
        v15 = *(__m128i **)(a4 + 8);
        v17.m128i_i64[0] = result;
        if ( *(__m128i **)(a4 + 16) != v15 )
        {
          if ( !v15 )
            goto LABEL_22;
          goto LABEL_21;
        }
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(&v17);
          v14 = *(__m128i **)(a4 + 8);
          result = *(_QWORD *)(a4 + 16);
        }
        v15 = v14 + 1;
        *(_QWORD *)(a4 + 8) = v15;
        v16 = *(_QWORD *)(a1 + 40);
        v17.m128i_i64[1] = a2 | 4;
        v17.m128i_i64[0] = v16;
        if ( v15 != (__m128i *)result )
        {
LABEL_21:
          *v15 = _mm_loadu_si128(&v17);
          v15 = *(__m128i **)(a4 + 8);
LABEL_22:
          *(_QWORD *)(a4 + 8) = v15 + 1;
          return result;
        }
      }
      return sub_F38BA0((const __m128i **)a4, v15, &v17);
    }
  }
  return result;
}
