// Function: sub_CE27D0
// Address: 0xce27d0
//
__int64 __fastcall sub_CE27D0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 result; // rax
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // r14d
  int v19; // eax
  __int64 v20; // rdx
  char v21; // dl
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rax
  const __m128i *v25; // rax
  const __m128i *v26; // rdi
  __m128i *v27; // rdx
  int v28; // ebx
  int v29; // eax
  __int64 v30; // [rsp+0h] [rbp-50h]
  unsigned __int64 v31[7]; // [rsp+18h] [rbp-38h] BYREF

LABEL_1:
  v2 = *(unsigned int *)(a1 + 104);
  v3 = *(_QWORD *)(a1 + 96);
  while ( 1 )
  {
    result = v3 + 40 * v2 - 40;
    v5 = *(_DWORD *)(result + 24);
    if ( *(_DWORD *)(result + 8) == v5 )
      return result;
    v6 = *(_QWORD *)(result + 16);
    *(_DWORD *)(result + 24) = v5 + 1;
    v9 = sub_B46EC0(v6, v5);
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_16;
    v12 = *(__int64 **)(a1 + 8);
    v8 = *(unsigned int *)(a1 + 20);
    v7 = &v12[v8];
    if ( v12 != v7 )
    {
      while ( v9 != *v12 )
      {
        if ( v7 == ++v12 )
          goto LABEL_7;
      }
      goto LABEL_1;
    }
LABEL_7:
    if ( (unsigned int)v8 < *(_DWORD *)(a1 + 16) )
    {
      *(_DWORD *)(a1 + 20) = v8 + 1;
      *v7 = v9;
      ++*(_QWORD *)a1;
    }
    else
    {
LABEL_16:
      sub_C8CC70(a1, v9, (__int64)v7, v8, v10, v11);
      if ( !v21 )
        goto LABEL_1;
    }
    v13 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v13 == v9 + 48 )
      goto LABEL_19;
    if ( !v13 )
      BUG();
    v14 = v13 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
    {
LABEL_19:
      v16 = *(unsigned int *)(a1 + 104);
      v18 = 0;
      v17 = 0;
      v14 = 0;
      v19 = v16;
      if ( *(_DWORD *)(a1 + 108) <= (unsigned int)v16 )
        goto LABEL_20;
LABEL_13:
      v3 = *(_QWORD *)(a1 + 96);
      v20 = v3 + 40 * v16;
      if ( v20 )
      {
        *(_QWORD *)v20 = v17;
        *(_DWORD *)(v20 + 8) = v18;
        *(_QWORD *)(v20 + 16) = v14;
        *(_DWORD *)(v20 + 24) = 0;
        *(_QWORD *)(v20 + 32) = v9;
        v19 = *(_DWORD *)(a1 + 104);
        v3 = *(_QWORD *)(a1 + 96);
      }
      v2 = (unsigned int)(v19 + 1);
      *(_DWORD *)(a1 + 104) = v2;
    }
    else
    {
      v15 = sub_B46E30(v14);
      v16 = *(unsigned int *)(a1 + 104);
      v17 = v14;
      v18 = v15;
      v19 = v16;
      if ( *(_DWORD *)(a1 + 108) > (unsigned int)v16 )
        goto LABEL_13;
LABEL_20:
      v30 = v17;
      v22 = a1 + 112;
      v3 = sub_C8D7D0(a1 + 96, a1 + 112, 0, 0x28u, v31, v17);
      v23 = 40LL * *(unsigned int *)(a1 + 104);
      v24 = v23 + v3;
      if ( v23 + v3 )
      {
        *(_DWORD *)(v24 + 8) = v18;
        *(_QWORD *)(v24 + 16) = v14;
        *(_QWORD *)v24 = v30;
        *(_DWORD *)(v24 + 24) = 0;
        *(_QWORD *)(v24 + 32) = v9;
        v23 = 40LL * *(unsigned int *)(a1 + 104);
      }
      v25 = *(const __m128i **)(a1 + 96);
      v26 = (const __m128i *)((char *)v25 + v23);
      if ( v25 != v26 )
      {
        v27 = (__m128i *)v3;
        do
        {
          if ( v27 )
          {
            *v27 = _mm_loadu_si128(v25);
            v27[1] = _mm_loadu_si128(v25 + 1);
            v27[2].m128i_i64[0] = v25[2].m128i_i64[0];
          }
          v25 = (const __m128i *)((char *)v25 + 40);
          v27 = (__m128i *)((char *)v27 + 40);
        }
        while ( v26 != v25 );
        v26 = *(const __m128i **)(a1 + 96);
      }
      v28 = v31[0];
      if ( (const __m128i *)v22 != v26 )
        _libc_free(v26, v22);
      v29 = *(_DWORD *)(a1 + 104);
      *(_QWORD *)(a1 + 96) = v3;
      *(_DWORD *)(a1 + 108) = v28;
      v2 = (unsigned int)(v29 + 1);
      *(_DWORD *)(a1 + 104) = v2;
    }
  }
}
