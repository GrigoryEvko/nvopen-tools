// Function: sub_37B2430
// Address: 0x37b2430
//
__int64 __fastcall sub_37B2430(__int64 a1, __m128i *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // r11
  __int32 v8; // edi
  int v9; // r13d
  __int64 result; // rax
  __int64 v11; // rdx
  int v12; // eax
  int v13; // r10d
  int v14; // edx
  __int64 v15; // rax
  __m128i v16; // xmm0
  int v17; // eax
  __m128i v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = v4 - 1;
    v7 = 0;
    v8 = a2->m128i_i32[2];
    v9 = 1;
    for ( result = (unsigned int)v6
                 & (v8
                  + ((unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)
                   ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 4))); ; result = (unsigned int)v6 & v12 )
    {
      v11 = v5 + 16LL * (unsigned int)result;
      if ( a2->m128i_i64[0] == *(_QWORD *)v11 && v8 == *(_DWORD *)(v11 + 8) )
        break;
      if ( !*(_QWORD *)v11 )
      {
        v13 = *(_DWORD *)(v11 + 8);
        if ( v13 == -1 )
        {
          v17 = *(_DWORD *)(a1 + 16);
          if ( !v7 )
            v7 = v11;
          ++*(_QWORD *)a1;
          v14 = v17 + 1;
          v19[0] = v7;
          if ( 4 * (v17 + 1) >= 3 * v4 )
            goto LABEL_14;
          if ( v4 - *(_DWORD *)(a1 + 20) - v14 > v4 >> 3 )
            goto LABEL_16;
          goto LABEL_15;
        }
        if ( v13 == -2 && !v7 )
          v7 = v5 + 16LL * (unsigned int)result;
      }
      v12 = v9 + result;
      ++v9;
    }
  }
  else
  {
    ++*(_QWORD *)a1;
    v19[0] = 0;
LABEL_14:
    v4 *= 2;
LABEL_15:
    sub_3437AD0(a1, v4);
    sub_3794520(a1, (unsigned __int64 *)a2, v19);
    v7 = v19[0];
    v14 = *(_DWORD *)(a1 + 16) + 1;
LABEL_16:
    *(_DWORD *)(a1 + 16) = v14;
    if ( *(_QWORD *)v7 || *(_DWORD *)(v7 + 8) != -1 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)v7 = a2->m128i_i64[0];
    *(_DWORD *)(v7 + 8) = a2->m128i_i32[2];
    v15 = *(unsigned int *)(a1 + 40);
    v16 = _mm_loadu_si128(a2);
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      v18 = v16;
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v15 + 1, 0x10u, v6, v5);
      v15 = *(unsigned int *)(a1 + 40);
      v16 = _mm_load_si128(&v18);
    }
    result = *(_QWORD *)(a1 + 32) + 16 * v15;
    *(__m128i *)result = v16;
    ++*(_DWORD *)(a1 + 40);
  }
  return result;
}
