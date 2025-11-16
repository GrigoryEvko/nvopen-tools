// Function: sub_146EFC0
// Address: 0x146efc0
//
__int64 __fastcall sub_146EFC0(__int64 a1, __m128i *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r11d
  __int64 *v7; // r10
  __int64 v8; // rdi
  __int64 v9; // r8
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // r8
  __int64 result; // rax
  __int64 *v13; // r8
  __int64 v14; // r13
  int v15; // eax
  int v16; // eax
  int v17; // edx
  __m128i *v18; // rsi
  __int64 *v19; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = a2->m128i_i64[1];
    v6 = 1;
    v7 = 0;
    v8 = *(_QWORD *)(a1 + 8);
    v9 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
    v10 = (((v9
           | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
          - 1
          - (v9 << 32)) >> 22)
        ^ ((v9
          | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
         - 1
         - (v9 << 32));
    v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
        ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
    for ( result = (v4 - 1) & ((unsigned int)((v11 - 1 - (v11 << 27)) >> 31) ^ ((_DWORD)v11 - 1 - ((_DWORD)v11 << 27)));
          ;
          result = (v4 - 1) & v15 )
    {
      v13 = (__int64 *)(v8 + 16LL * (unsigned int)result);
      v14 = *v13;
      if ( *v13 == a2->m128i_i64[0] && v13[1] == v5 )
        break;
      if ( v14 == -8 )
      {
        if ( v13[1] == -8 )
        {
          v16 = *(_DWORD *)(a1 + 16);
          if ( !v7 )
            v7 = v13;
          ++*(_QWORD *)a1;
          v17 = v16 + 1;
          if ( 4 * (v16 + 1) >= 3 * v4 )
            goto LABEL_26;
          if ( v4 - *(_DWORD *)(a1 + 20) - v17 <= v4 >> 3 )
            goto LABEL_27;
          goto LABEL_17;
        }
      }
      else if ( v14 == -16 && v13[1] == -16 && !v7 )
      {
        v7 = (__int64 *)(v8 + 16LL * (unsigned int)result);
      }
      v15 = v6 + result;
      ++v6;
    }
  }
  else
  {
    ++*(_QWORD *)a1;
LABEL_26:
    v4 *= 2;
LABEL_27:
    sub_146ED20(a1, v4);
    sub_14640F0(a1, a2->m128i_i64, &v19);
    v7 = v19;
    v17 = *(_DWORD *)(a1 + 16) + 1;
LABEL_17:
    *(_DWORD *)(a1 + 16) = v17;
    if ( *v7 != -8 || v7[1] != -8 )
      --*(_DWORD *)(a1 + 20);
    *v7 = a2->m128i_i64[0];
    result = a2->m128i_i64[1];
    v7[1] = result;
    v18 = *(__m128i **)(a1 + 40);
    if ( v18 == *(__m128i **)(a1 + 48) )
    {
      return sub_145F860((const __m128i **)(a1 + 32), v18, a2);
    }
    else
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(a2);
        v18 = *(__m128i **)(a1 + 40);
      }
      *(_QWORD *)(a1 + 40) = v18 + 1;
    }
  }
  return result;
}
