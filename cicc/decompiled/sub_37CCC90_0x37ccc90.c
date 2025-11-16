// Function: sub_37CCC90
// Address: 0x37ccc90
//
__int64 __fastcall sub_37CCC90(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 i; // rdx
  __int64 v10; // rbx
  __int64 *v11; // rdx
  __int64 v12; // rsi
  __m128i *v13; // rax
  __int64 j; // rdx
  __int64 *v15; // [rsp+8h] [rbp-48h]
  __m128i *v16; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_C7D670(48LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = 48 * v2;
    v8 = v3 + 48 * v2;
    for ( i = result + 48 * v6; i != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
      }
    }
    if ( v8 != v3 )
    {
      v10 = v3;
      v11 = (__int64 *)&v16;
      do
      {
        while ( *(_QWORD *)v10
             || *(_BYTE *)(v10 + 24) && (*(_QWORD *)(v10 + 8) || *(_QWORD *)(v10 + 16))
             || *(_QWORD *)(v10 + 32) )
        {
          v12 = v10;
          v15 = v11;
          v10 += 48;
          sub_37BE800(a1, v12, v11);
          v13 = v16;
          v11 = v15;
          *v16 = _mm_loadu_si128((const __m128i *)(v10 - 48));
          v13[1] = _mm_loadu_si128((const __m128i *)(v10 - 32));
          v13[2].m128i_i64[0] = *(_QWORD *)(v10 - 16);
          v16[2].m128i_i32[2] = *(_DWORD *)(v10 - 8);
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return sub_C7D6A0(v3, v7, 8);
        }
        v10 += 48;
      }
      while ( v8 != v10 );
    }
    return sub_C7D6A0(v3, v7, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 48LL * *(unsigned int *)(a1 + 24); j != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
      }
    }
  }
  return result;
}
