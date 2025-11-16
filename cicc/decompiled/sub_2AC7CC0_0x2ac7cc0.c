// Function: sub_2AC7CC0
// Address: 0x2ac7cc0
//
__int64 __fastcall sub_2AC7CC0(__int64 a1, int a2)
{
  __m128i *v2; // rbx
  __int64 v3; // r14
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __m128i *v8; // r12
  __int64 i; // rdx
  __m128i *v10; // rax
  __m128i *v11; // rax
  __int64 v12; // rdx
  __int64 j; // rdx
  __int64 v14; // [rsp+8h] [rbp-48h]
  __m128i *v15; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(__m128i **)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 24);
  v14 = (__int64)v2;
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_C7D670((unsigned __int64)v4 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v2 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    v7 = v3 << 6;
    *(_QWORD *)(a1 + 16) = 0;
    v8 = (__m128i *)((char *)v2 + v7);
    for ( i = result + (v6 << 6); i != result; result += 64 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 1;
      }
    }
    if ( v8 != v2 )
    {
      while ( v2->m128i_i64[0] == -4096 )
      {
        if ( v2->m128i_i32[2] == -1 && v2->m128i_i8[12] )
        {
          v2 += 4;
          if ( v8 == v2 )
            return sub_C7D6A0(v14, v7, 8);
        }
        else
        {
LABEL_11:
          sub_2ABE520(a1, v2->m128i_i64, &v15);
          v10 = v15;
          v15->m128i_i64[0] = v2->m128i_i64[0];
          v10->m128i_i32[2] = v2->m128i_i32[2];
          v10->m128i_i8[12] = v2->m128i_i8[12];
          v11 = v15;
          v15[1] = _mm_loadu_si128(v2 + 1);
          v11[2] = _mm_loadu_si128(v2 + 2);
          v11[3] = _mm_loadu_si128(v2 + 3);
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v2 += 4;
          if ( v8 == v2 )
            return sub_C7D6A0(v14, v7, 8);
        }
      }
      if ( v2->m128i_i64[0] == -8192 && v2->m128i_i32[2] == -2 && !v2->m128i_i8[12] )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v14, v7, 8);
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + (v12 << 6); j != result; result += 64 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 1;
      }
    }
  }
  return result;
}
