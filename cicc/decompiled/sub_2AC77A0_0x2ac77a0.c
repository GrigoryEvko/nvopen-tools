// Function: sub_2AC77A0
// Address: 0x2ac77a0
//
__int64 __fastcall sub_2AC77A0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // eax
  __int64 result; // rax
  __m128i *v6; // r12
  __int64 i; // rdx
  __m128i *v8; // rbx
  __m128i *v9; // rax
  __m128i *v10; // rax
  __int64 v11; // rdx
  __int64 j; // rdx
  __int64 v13; // [rsp+8h] [rbp-48h]
  __m128i *v14; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_C7D670(40LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v13 = 40 * v2;
    v6 = (__m128i *)(v3 + 40 * v2);
    for ( i = result + 40LL * *(unsigned int *)(a1 + 24); i != result; result += 40 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = -1;
        *(_BYTE *)(result + 12) = 1;
      }
    }
    if ( v6 != (__m128i *)v3 )
    {
      v8 = (__m128i *)v3;
      while ( v8->m128i_i64[0] == -4096 )
      {
        if ( v8->m128i_i32[2] == -1 && v8->m128i_i8[12] )
        {
          v8 = (__m128i *)((char *)v8 + 40);
          if ( v6 == v8 )
            return sub_C7D6A0(v3, v13, 8);
        }
        else
        {
LABEL_11:
          sub_2ABE410(a1, v8->m128i_i64, &v14);
          v9 = v14;
          v14->m128i_i64[0] = v8->m128i_i64[0];
          v9->m128i_i32[2] = v8->m128i_i32[2];
          v9->m128i_i8[12] = v8->m128i_i8[12];
          v10 = v14;
          v14[1] = _mm_loadu_si128(v8 + 1);
          v10[2].m128i_i64[0] = v8[2].m128i_i64[0];
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v8 = (__m128i *)((char *)v8 + 40);
          if ( v6 == v8 )
            return sub_C7D6A0(v3, v13, 8);
        }
      }
      if ( v8->m128i_i64[0] == -8192 && v8->m128i_i32[2] == -2 && !v8->m128i_i8[12] )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v3, v13, 8);
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 40 * v11; j != result; result += 40 )
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
