// Function: sub_1C35DC0
// Address: 0x1c35dc0
//
__int64 __fastcall sub_1C35DC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r11d
  __int64 *v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 result; // rax
  int v12; // eax
  int v13; // edx
  __int64 v14; // r12
  __int64 v15; // r14
  unsigned __int8 *v16; // rsi
  int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rdi
  _WORD *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdi
  __m128i *v26; // rdx
  __m128i si128; // xmm0
  _BYTE *v28; // rax
  int v29; // eax
  int v30; // ecx
  __int64 v31; // r8
  unsigned int v32; // eax
  __int64 v33; // rsi
  int v34; // r10d
  __int64 *v35; // r9
  int v36; // eax
  int v37; // eax
  __int64 v38; // rsi
  int v39; // r9d
  __int64 *v40; // r8
  unsigned int v41; // r12d
  __int64 v42; // rcx

  v2 = a1 + 144;
  v5 = *(_DWORD *)(a1 + 168);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 144);
    goto LABEL_39;
  }
  v6 = *(_QWORD *)(a1 + 152);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 8LL * v9);
  result = *v10;
  if ( a2 == *v10 )
    return result;
  while ( result != -8 )
  {
    if ( v8 || result != -16 )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    result = *(_QWORD *)(v6 + 8LL * v9);
    if ( a2 == result )
      return result;
    ++v7;
    v8 = v10;
    v10 = (__int64 *)(v6 + 8LL * v9);
  }
  v12 = *(_DWORD *)(a1 + 160);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)(a1 + 144);
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_39:
    sub_1647990(v2, 2 * v5);
    v29 = *(_DWORD *)(a1 + 168);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 152);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v31 + 8LL * v32);
      v33 = *v8;
      v13 = *(_DWORD *)(a1 + 160) + 1;
      if ( a2 != *v8 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -8 )
        {
          if ( !v35 && v33 == -16 )
            v35 = v8;
          v32 = v30 & (v34 + v32);
          v8 = (__int64 *)(v31 + 8LL * v32);
          v33 = *v8;
          if ( a2 == *v8 )
            goto LABEL_13;
          ++v34;
        }
        if ( v35 )
          v8 = v35;
      }
      goto LABEL_13;
    }
    goto LABEL_68;
  }
  if ( v5 - *(_DWORD *)(a1 + 164) - v13 <= v5 >> 3 )
  {
    sub_1647990(v2, v5);
    v36 = *(_DWORD *)(a1 + 168);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 152);
      v39 = 1;
      v40 = 0;
      v41 = v37 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v38 + 8LL * v41);
      v42 = *v8;
      v13 = *(_DWORD *)(a1 + 160) + 1;
      if ( a2 != *v8 )
      {
        while ( v42 != -8 )
        {
          if ( v42 == -16 && !v40 )
            v40 = v8;
          v41 = v37 & (v39 + v41);
          v8 = (__int64 *)(v38 + 8LL * v41);
          v42 = *v8;
          if ( a2 == *v8 )
            goto LABEL_13;
          ++v39;
        }
        if ( v40 )
          v8 = v40;
      }
      goto LABEL_13;
    }
LABEL_68:
    ++*(_DWORD *)(a1 + 160);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 160) = v13;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 164);
  *v8 = a2;
  result = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)result )
  {
    v14 = 0;
    v15 = *(unsigned int *)(a2 + 8);
    v16 = *(unsigned __int8 **)(a2 - 8 * result);
    if ( !v16 )
      goto LABEL_20;
    v17 = *v16;
    result = (unsigned int)(v17 - 1);
    if ( (unsigned int)result > 1 )
      goto LABEL_23;
LABEL_18:
    v18 = *((_QWORD *)v16 + 17);
    if ( v18 )
      result = sub_1C34B00(a1, v18);
LABEL_20:
    while ( v15 != ++v14 )
    {
      result = *(unsigned int *)(a2 + 8);
      v16 = *(unsigned __int8 **)(a2 + 8 * (v14 - result));
      if ( v16 )
      {
        v17 = *v16;
        result = (unsigned int)(v17 - 1);
        if ( (unsigned int)result <= 1 )
          goto LABEL_18;
LABEL_23:
        result = (unsigned int)(v17 - 4);
        if ( (unsigned __int8)(v17 - 4) > 0x1Eu )
        {
          if ( (_BYTE)v17 != 3 && (_BYTE)v17 )
          {
            v19 = *(_QWORD *)(a1 + 24);
            v20 = *(_QWORD *)(v19 + 24);
            if ( (unsigned __int64)(*(_QWORD *)(v19 + 16) - v20) <= 6 )
            {
              sub_16E7EE0(v19, "Error: ", 7u);
            }
            else
            {
              *(_DWORD *)v20 = 1869771333;
              *(_WORD *)(v20 + 4) = 14962;
              *(_BYTE *)(v20 + 6) = 32;
              *(_QWORD *)(v19 + 24) += 7LL;
            }
            v21 = *(_QWORD *)(a1 + 24);
            v22 = *(_WORD **)(v21 + 24);
            if ( *(_QWORD *)(v21 + 16) - (_QWORD)v22 <= 1u )
            {
              sub_16E7EE0(v21, ": ", 2u);
            }
            else
            {
              *v22 = 8250;
              *(_QWORD *)(v21 + 24) += 2LL;
            }
            sub_1556280((unsigned __int8 *)a2, *(_QWORD *)(a1 + 24), 0);
            v23 = *(_QWORD *)(a1 + 24);
            v24 = *(_QWORD *)(v23 + 24);
            if ( (unsigned __int64)(*(_QWORD *)(v23 + 16) - v24) <= 2 )
            {
              sub_16E7EE0(v23, "\n  ", 3u);
            }
            else
            {
              *(_BYTE *)(v24 + 2) = 32;
              *(_WORD *)v24 = 8202;
              *(_QWORD *)(v23 + 24) += 3LL;
            }
            v25 = *(_QWORD *)(a1 + 24);
            v26 = *(__m128i **)(v25 + 24);
            if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 0x14u )
            {
              v25 = sub_16E7EE0(v25, "Invalid metadata type", 0x15u);
              v28 = *(_BYTE **)(v25 + 24);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_42D0AE0);
              v26[1].m128i_i32[0] = 1887007776;
              v26[1].m128i_i8[4] = 101;
              *v26 = si128;
              v28 = (_BYTE *)(*(_QWORD *)(v25 + 24) + 21LL);
              *(_QWORD *)(v25 + 24) = v28;
            }
            if ( v28 == *(_BYTE **)(v25 + 16) )
            {
              sub_16E7EE0(v25, "\n", 1u);
            }
            else
            {
              *v28 = 10;
              ++*(_QWORD *)(v25 + 24);
            }
            result = sub_1C31880(a1);
          }
        }
        else
        {
          result = sub_1C35DC0(a1);
        }
      }
    }
  }
  return result;
}
