// Function: sub_37C5160
// Address: 0x37c5160
//
__int64 __fastcall sub_37C5160(__int64 a1, int a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __m128i *v8; // r12
  __int64 i; // rcx
  int v10; // edx
  __m128i *v11; // r15
  __int8 v12; // al
  int *v13; // rsi
  __m128i *v14; // rax
  __int64 v15; // rdx
  __int64 j; // rcx
  int v17; // edx
  __m128i *v18; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v19[6]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v20[12]; // [rsp+50h] [rbp-60h] BYREF

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
    v8 = (__m128i *)(v3 + 48 * v2);
    for ( i = result + 48 * v6; i != result; result += 48 )
    {
      if ( result )
      {
        v10 = *(_DWORD *)result;
        *(_QWORD *)(result + 16) = 0;
        *(_DWORD *)result = v10 & 0xFFF00000 | 0x15;
      }
    }
    v19[0] = 21;
    v19[2] = 0;
    v20[0] = 22;
    v20[2] = 0;
    if ( v8 != (__m128i *)v3 )
    {
      v11 = (__m128i *)v3;
      while ( 1 )
      {
        v12 = v11->m128i_i8[0];
        if ( (unsigned __int8)(v11->m128i_i8[0] - 21) <= 1u )
          break;
        if ( (unsigned __int8)sub_2EAB6C0((__int64)v11, (char *)v19) )
          goto LABEL_11;
        v12 = v11->m128i_i8[0];
        if ( (unsigned __int8)(v11->m128i_i8[0] - 21) <= 1u )
        {
LABEL_18:
          if ( v12 != LOBYTE(v20[0]) )
            goto LABEL_16;
LABEL_11:
          v11 += 3;
          if ( v8 == v11 )
            return sub_C7D6A0(v3, v7, 8);
        }
        else
        {
          if ( (unsigned __int8)sub_2EAB6C0((__int64)v11, (char *)v20) )
            goto LABEL_11;
LABEL_16:
          v13 = (int *)v11;
          v11 += 3;
          sub_37BD360(a1, v13, &v18);
          v14 = v18;
          *v18 = _mm_loadu_si128(v11 - 3);
          v14[1] = _mm_loadu_si128(v11 - 2);
          v14[2].m128i_i64[0] = v11[-1].m128i_i64[0];
          v18[2].m128i_i32[2] = v11[-1].m128i_i32[2];
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v11 )
            return sub_C7D6A0(v3, v7, 8);
        }
      }
      if ( v12 == LOBYTE(v19[0]) )
        goto LABEL_11;
      goto LABEL_18;
    }
    return sub_C7D6A0(v3, v7, 8);
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 48 * v15; j != result; result += 48 )
    {
      if ( result )
      {
        v17 = *(_DWORD *)result;
        *(_QWORD *)(result + 16) = 0;
        *(_DWORD *)result = v17 & 0xFFF00000 | 0x15;
      }
    }
  }
  return result;
}
