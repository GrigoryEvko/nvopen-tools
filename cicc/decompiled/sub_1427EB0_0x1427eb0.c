// Function: sub_1427EB0
// Address: 0x1427eb0
//
_QWORD *__fastcall sub_1427EB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rdi
  unsigned int v13; // ecx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rbx
  int v18; // esi
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // ecx
  int v22; // edx
  __int64 v23; // rdi
  _QWORD *result; // rax
  _QWORD *v25; // r11
  int v26; // edi
  int v27; // ecx
  int v28; // ecx
  _QWORD *v29; // r11
  int v30; // r8d
  unsigned int v31; // r15d
  __int64 v32; // rdi
  __int64 v33; // rsi
  _QWORD *v34; // rsi
  unsigned int v35; // edi
  _QWORD *v36; // rcx
  int v37; // edx
  int v38; // r9d
  int v39; // r11d
  _QWORD *v40; // r15
  __int64 v41; // [rsp+8h] [rbp-48h]
  int v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+10h] [rbp-40h]
  __int64 v45; // [rsp+18h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 80);
  if ( !(_DWORD)v2 )
    goto LABEL_67;
  v5 = *(_QWORD *)(a1 + 64);
  v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v37 = 1;
    while ( v8 != -8 )
    {
      v38 = v37 + 1;
      v6 = (v2 - 1) & (v37 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v37 = v38;
    }
LABEL_67:
    BUG();
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 16 * v2) )
    goto LABEL_67;
  v9 = v7[1];
  v10 = 0;
  v44 = a1 + 296;
  v11 = *(_QWORD *)(v9 + 8);
  if ( v9 == v11 )
    goto LABEL_18;
  v45 = a2;
  do
  {
    while ( 1 )
    {
      v16 = *(_DWORD *)(a1 + 320);
      v17 = v11 - 32;
      if ( !v11 )
        v17 = 0;
      ++v10;
      if ( !v16 )
      {
        ++*(_QWORD *)(a1 + 296);
LABEL_12:
        v41 = v9;
        sub_1427CF0(v44, 2 * v16);
        v18 = *(_DWORD *)(a1 + 320);
        if ( !v18 )
          goto LABEL_68;
        v19 = v18 - 1;
        v20 = *(_QWORD *)(a1 + 304);
        v9 = v41;
        v21 = v19 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v22 = *(_DWORD *)(a1 + 312) + 1;
        v14 = (_QWORD *)(v20 + 16LL * v21);
        v23 = *v14;
        if ( v17 != *v14 )
        {
          v39 = 1;
          v40 = 0;
          while ( v23 != -8 )
          {
            if ( v23 == -16 && !v40 )
              v40 = v14;
            v21 = v19 & (v39 + v21);
            v14 = (_QWORD *)(v20 + 16LL * v21);
            v23 = *v14;
            if ( v17 == *v14 )
              goto LABEL_14;
            ++v39;
          }
          if ( v40 )
            v14 = v40;
        }
        goto LABEL_14;
      }
      v12 = *(_QWORD *)(a1 + 304);
      v13 = (v16 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v14 = (_QWORD *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v17 != *v14 )
        break;
LABEL_7:
      v14[1] = v10;
      v11 = *(_QWORD *)(v11 + 8);
      if ( v9 == v11 )
        goto LABEL_17;
    }
    v42 = 1;
    v25 = 0;
    while ( v15 != -8 )
    {
      if ( !v25 && v15 == -16 )
        v25 = v14;
      v13 = (v16 - 1) & (v42 + v13);
      v14 = (_QWORD *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v17 == *v14 )
        goto LABEL_7;
      ++v42;
    }
    v26 = *(_DWORD *)(a1 + 312);
    if ( v25 )
      v14 = v25;
    ++*(_QWORD *)(a1 + 296);
    v22 = v26 + 1;
    if ( 4 * (v26 + 1) >= 3 * v16 )
      goto LABEL_12;
    if ( v16 - *(_DWORD *)(a1 + 316) - v22 <= v16 >> 3 )
    {
      v43 = v9;
      sub_1427CF0(v44, v16);
      v27 = *(_DWORD *)(a1 + 320);
      if ( !v27 )
      {
LABEL_68:
        ++*(_DWORD *)(a1 + 312);
        BUG();
      }
      v28 = v27 - 1;
      v29 = 0;
      v9 = v43;
      v30 = 1;
      v31 = v28 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v32 = *(_QWORD *)(a1 + 304);
      v22 = *(_DWORD *)(a1 + 312) + 1;
      v14 = (_QWORD *)(v32 + 16LL * v31);
      v33 = *v14;
      if ( v17 != *v14 )
      {
        while ( v33 != -8 )
        {
          if ( !v29 && v33 == -16 )
            v29 = v14;
          v31 = v28 & (v30 + v31);
          v14 = (_QWORD *)(v32 + 16LL * v31);
          v33 = *v14;
          if ( v17 == *v14 )
            goto LABEL_14;
          ++v30;
        }
        if ( v29 )
          v14 = v29;
      }
    }
LABEL_14:
    *(_DWORD *)(a1 + 312) = v22;
    if ( *v14 != -8 )
      --*(_DWORD *)(a1 + 316);
    v14[1] = 0;
    *v14 = v17;
    v14[1] = v10;
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v9 != v11 );
LABEL_17:
  a2 = v45;
LABEL_18:
  result = *(_QWORD **)(a1 + 136);
  if ( *(_QWORD **)(a1 + 144) != result )
    return (_QWORD *)sub_16CCBA0(a1 + 128, a2);
  v34 = &result[*(unsigned int *)(a1 + 156)];
  v35 = *(_DWORD *)(a1 + 156);
  if ( result == v34 )
  {
LABEL_46:
    if ( v35 >= *(_DWORD *)(a1 + 152) )
      return (_QWORD *)sub_16CCBA0(a1 + 128, a2);
    *(_DWORD *)(a1 + 156) = v35 + 1;
    *v34 = a2;
    ++*(_QWORD *)(a1 + 128);
  }
  else
  {
    v36 = 0;
    while ( a2 != *result )
    {
      if ( *result == -2 )
        v36 = result;
      if ( v34 == ++result )
      {
        if ( !v36 )
          goto LABEL_46;
        *v36 = a2;
        --*(_DWORD *)(a1 + 160);
        ++*(_QWORD *)(a1 + 128);
        return result;
      }
    }
  }
  return result;
}
