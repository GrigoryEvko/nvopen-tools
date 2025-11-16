// Function: sub_1072BB0
// Address: 0x1072bb0
//
__int64 __fastcall sub_1072BB0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // r15
  __int64 result; // rax
  int v7; // r14d
  unsigned int v8; // r8d
  __int64 v9; // r9
  _QWORD *v10; // rdi
  unsigned int v11; // ecx
  int v12; // r11d
  __int64 v13; // rdx
  __m128i *v14; // rdx
  const __m128i *v15; // rsi
  const __m128i **v16; // rdi
  unsigned int v17; // esi
  int v18; // edx
  int v19; // edx
  __int64 v20; // r9
  unsigned int v21; // ecx
  int v22; // eax
  __int64 v23; // rsi
  int v24; // r11d
  _QWORD *v25; // r10
  int v26; // eax
  int v27; // edx
  int v28; // edx
  __int64 v29; // r9
  int v30; // r11d
  unsigned int v31; // ecx
  __int64 v32; // rsi
  __int64 v33; // [rsp+0h] [rbp-50h]
  unsigned int v34; // [rsp+Ch] [rbp-44h]
  unsigned int v35; // [rsp+Ch] [rbp-44h]
  unsigned int v36; // [rsp+Ch] [rbp-44h]
  __m128i v37; // [rsp+10h] [rbp-40h] BYREF

  v2 = *(__int64 **)(a1 + 56);
  v3 = *(__int64 **)(a1 + 64);
  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 168LL) + 680LL);
  result = *(_QWORD *)(a1 + 104);
  if ( v2 != v3 )
  {
    v7 = (((*(_BYTE *)(result + 8) & 1) != 0) + 2) << 25;
    v8 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
    v33 = a1 + 112;
    while ( 1 )
    {
      result = *v2;
      if ( (*(_BYTE *)(*v2 + 8) & 0x10) != 0 )
        break;
LABEL_8:
      if ( v3 == ++v2 )
        return result;
    }
    v17 = *(_DWORD *)(a1 + 136);
    v37.m128i_i64[0] = *v2;
    v37.m128i_i32[2] = 0;
    v37.m128i_i32[3] = v7;
    if ( v17 )
    {
      v9 = *(_QWORD *)(a1 + 120);
      v10 = 0;
      v11 = (v17 - 1) & v8;
      v12 = 1;
      result = v9 + 32LL * v11;
      v13 = *(_QWORD *)result;
      if ( v4 == *(_QWORD *)result )
      {
LABEL_4:
        v14 = *(__m128i **)(result + 16);
        v15 = *(const __m128i **)(result + 24);
        v16 = (const __m128i **)(result + 8);
        if ( v14 != v15 )
        {
          if ( v14 )
          {
            *v14 = _mm_loadu_si128(&v37);
            v14 = *(__m128i **)(result + 16);
          }
          *(_QWORD *)(result + 16) = v14 + 1;
          goto LABEL_8;
        }
        goto LABEL_33;
      }
      while ( v13 != -4096 )
      {
        if ( !v10 && v13 == -8192 )
          v10 = (_QWORD *)result;
        v11 = (v17 - 1) & (v12 + v11);
        result = v9 + 32LL * v11;
        v13 = *(_QWORD *)result;
        if ( v4 == *(_QWORD *)result )
          goto LABEL_4;
        ++v12;
      }
      if ( !v10 )
        v10 = (_QWORD *)result;
      v26 = *(_DWORD *)(a1 + 128);
      ++*(_QWORD *)(a1 + 112);
      v22 = v26 + 1;
      if ( 4 * v22 < 3 * v17 )
      {
        if ( v17 - *(_DWORD *)(a1 + 132) - v22 > v17 >> 3 )
          goto LABEL_30;
        v36 = v8;
        sub_1072980(v33, v17);
        v27 = *(_DWORD *)(a1 + 136);
        if ( !v27 )
        {
LABEL_47:
          ++*(_DWORD *)(a1 + 128);
          BUG();
        }
        v8 = v36;
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 120);
        v25 = 0;
        v30 = 1;
        v31 = v28 & v36;
        v22 = *(_DWORD *)(a1 + 128) + 1;
        v10 = (_QWORD *)(v29 + 32LL * (v28 & v36));
        v32 = *v10;
        if ( v4 == *v10 )
          goto LABEL_30;
        while ( v32 != -4096 )
        {
          if ( !v25 && v32 == -8192 )
            v25 = v10;
          v31 = v28 & (v30 + v31);
          v10 = (_QWORD *)(v29 + 32LL * v31);
          v32 = *v10;
          if ( v4 == *v10 )
            goto LABEL_30;
          ++v30;
        }
        goto LABEL_16;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 112);
    }
    v34 = v8;
    sub_1072980(v33, 2 * v17);
    v18 = *(_DWORD *)(a1 + 136);
    if ( !v18 )
      goto LABEL_47;
    v8 = v34;
    v19 = v18 - 1;
    v20 = *(_QWORD *)(a1 + 120);
    v21 = v19 & v34;
    v22 = *(_DWORD *)(a1 + 128) + 1;
    v10 = (_QWORD *)(v20 + 32LL * (v19 & v34));
    v23 = *v10;
    if ( v4 == *v10 )
      goto LABEL_30;
    v24 = 1;
    v25 = 0;
    while ( v23 != -4096 )
    {
      if ( !v25 && v23 == -8192 )
        v25 = v10;
      v21 = v19 & (v24 + v21);
      v10 = (_QWORD *)(v20 + 32LL * v21);
      v23 = *v10;
      if ( v4 == *v10 )
        goto LABEL_30;
      ++v24;
    }
LABEL_16:
    if ( v25 )
      v10 = v25;
LABEL_30:
    *(_DWORD *)(a1 + 128) = v22;
    if ( *v10 != -4096 )
      --*(_DWORD *)(a1 + 132);
    *v10 = v4;
    v15 = 0;
    v16 = (const __m128i **)(v10 + 1);
    *v16 = 0;
    v16[1] = 0;
    v16[2] = 0;
LABEL_33:
    v35 = v8;
    result = sub_1072660(v16, v15, &v37);
    v8 = v35;
    goto LABEL_8;
  }
  return result;
}
