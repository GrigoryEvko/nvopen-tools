// Function: sub_1E66BB0
// Address: 0x1e66bb0
//
_QWORD *__fastcall sub_1E66BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // r8d
  unsigned int v9; // eax
  __int64 *v10; // r9
  __int64 v11; // rcx
  unsigned int v12; // r13d
  unsigned int v13; // r10d
  _QWORD *result; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  _QWORD *v17; // rcx
  int v18; // r9d
  int v19; // eax
  int v20; // esi
  __int64 v21; // r9
  unsigned int v22; // ecx
  int v23; // edi
  __int64 v24; // r8
  int v25; // r9d
  unsigned int v26; // r13d
  unsigned int v27; // r9d
  __int64 v28; // rcx
  int v29; // r10d
  int v30; // edi
  int v31; // ecx
  int v32; // eax
  int v33; // edx
  __int64 v34; // rdi
  unsigned int v35; // r13d
  __int64 v36; // rsi
  int v37; // r9d
  _QWORD *v38; // r8
  int v39; // eax
  int v40; // edx
  __int64 v41; // rdi
  int v42; // r9d
  unsigned int v43; // r13d
  __int64 v44; // rsi
  int v45; // r11d
  _QWORD *v46; // r10
  int v47; // edi
  int v48; // eax
  int v49; // ecx
  __int64 v50; // r8
  _QWORD *v51; // r9
  unsigned int v52; // r13d
  int v53; // r10d
  __int64 v54; // rsi
  int v55; // r11d
  _QWORD *v56; // r10
  __int64 v57; // [rsp+8h] [rbp-28h]
  __int64 v58; // [rsp+8h] [rbp-28h]

  v6 = *(_DWORD *)(a4 + 24);
  v7 = *(_QWORD *)(a4 + 8);
  if ( !v6 )
  {
    ++*(_QWORD *)a4;
    goto LABEL_7;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a3 == *v10 )
  {
LABEL_3:
    v12 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v13 = v12 & v8;
    result = (_QWORD *)(v7 + 16LL * (v12 & v8));
    if ( v10 != (__int64 *)(v7 + 16LL * v6) )
    {
      v15 = *result;
      v16 = v10[1];
      v17 = 0;
      v18 = 1;
      if ( *result == a2 )
      {
LABEL_5:
        result[1] = v16;
        return result;
      }
      while ( v15 != -8 )
      {
        if ( !v17 && v15 == -16 )
          v17 = result;
        v13 = v8 & (v18 + v13);
        result = (_QWORD *)(v7 + 16LL * v13);
        v15 = *result;
        if ( *result == a2 )
          goto LABEL_5;
        ++v18;
      }
      v30 = *(_DWORD *)(a4 + 16);
      if ( v17 )
        result = v17;
      ++*(_QWORD *)a4;
      v31 = v30 + 1;
      if ( 4 * (v30 + 1) >= 3 * v6 )
      {
        sub_1E669F0(a4, 2 * v6);
        v32 = *(_DWORD *)(a4 + 24);
        if ( v32 )
        {
          v33 = v32 - 1;
          v34 = *(_QWORD *)(a4 + 8);
          v35 = (v32 - 1) & v12;
          v31 = *(_DWORD *)(a4 + 16) + 1;
          result = (_QWORD *)(v34 + 16LL * v35);
          v36 = *result;
          if ( *result == a2 )
            goto LABEL_28;
          v37 = 1;
          v38 = 0;
          while ( v36 != -8 )
          {
            if ( v36 == -16 && !v38 )
              v38 = result;
            v35 = v33 & (v37 + v35);
            result = (_QWORD *)(v34 + 16LL * v35);
            v36 = *result;
            if ( *result == a2 )
              goto LABEL_28;
            ++v37;
          }
LABEL_35:
          if ( v38 )
            result = v38;
          goto LABEL_28;
        }
      }
      else
      {
        if ( v6 - *(_DWORD *)(a4 + 20) - v31 > v6 >> 3 )
        {
LABEL_28:
          *(_DWORD *)(a4 + 16) = v31;
          if ( *result != -8 )
            --*(_DWORD *)(a4 + 20);
          *result = a2;
          result[1] = 0;
          goto LABEL_5;
        }
        sub_1E669F0(a4, v6);
        v39 = *(_DWORD *)(a4 + 24);
        if ( v39 )
        {
          v40 = v39 - 1;
          v41 = *(_QWORD *)(a4 + 8);
          v42 = 1;
          v43 = (v39 - 1) & v12;
          v38 = 0;
          v31 = *(_DWORD *)(a4 + 16) + 1;
          result = (_QWORD *)(v41 + 16LL * v43);
          v44 = *result;
          if ( *result == a2 )
            goto LABEL_28;
          while ( v44 != -8 )
          {
            if ( v44 == -16 && !v38 )
              v38 = result;
            v43 = v40 & (v42 + v43);
            result = (_QWORD *)(v41 + 16LL * v43);
            v44 = *result;
            if ( *result == a2 )
              goto LABEL_28;
            ++v42;
          }
          goto LABEL_35;
        }
      }
LABEL_83:
      ++*(_DWORD *)(a4 + 16);
      BUG();
    }
  }
  else
  {
    v25 = 1;
    while ( v11 != -8 )
    {
      v29 = v25 + 1;
      v9 = v8 & (v25 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a3 == *v10 )
        goto LABEL_3;
      v25 = v29;
    }
  }
  v26 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v27 = v26 & v8;
  result = (_QWORD *)(v7 + 16LL * (v26 & v8));
  v28 = *result;
  if ( *result != a2 )
  {
    v45 = 1;
    v46 = 0;
    while ( v28 != -8 )
    {
      if ( !v46 && v28 == -16 )
        v46 = result;
      v27 = v8 & (v45 + v27);
      result = (_QWORD *)(v7 + 16LL * v27);
      v28 = *result;
      if ( *result == a2 )
        goto LABEL_15;
      ++v45;
    }
    v47 = *(_DWORD *)(a4 + 16);
    if ( v46 )
      result = v46;
    ++*(_QWORD *)a4;
    v23 = v47 + 1;
    if ( 4 * v23 < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a4 + 20) - v23 <= v6 >> 3 )
      {
        v58 = a3;
        sub_1E669F0(a4, v6);
        v48 = *(_DWORD *)(a4 + 24);
        if ( !v48 )
          goto LABEL_83;
        v49 = v48 - 1;
        v50 = *(_QWORD *)(a4 + 8);
        v51 = 0;
        v52 = (v48 - 1) & v26;
        a3 = v58;
        v53 = 1;
        v23 = *(_DWORD *)(a4 + 16) + 1;
        result = (_QWORD *)(v50 + 16LL * v52);
        v54 = *result;
        if ( *result != a2 )
        {
          while ( v54 != -8 )
          {
            if ( !v51 && v54 == -16 )
              v51 = result;
            v52 = v49 & (v53 + v52);
            result = (_QWORD *)(v50 + 16LL * v52);
            v54 = *result;
            if ( *result == a2 )
              goto LABEL_9;
            ++v53;
          }
          if ( v51 )
            result = v51;
        }
      }
      goto LABEL_9;
    }
LABEL_7:
    v57 = a3;
    sub_1E669F0(a4, 2 * v6);
    v19 = *(_DWORD *)(a4 + 24);
    if ( !v19 )
      goto LABEL_83;
    v20 = v19 - 1;
    v21 = *(_QWORD *)(a4 + 8);
    a3 = v57;
    v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = *(_DWORD *)(a4 + 16) + 1;
    result = (_QWORD *)(v21 + 16LL * v22);
    v24 = *result;
    if ( *result != a2 )
    {
      v55 = 1;
      v56 = 0;
      while ( v24 != -8 )
      {
        if ( v24 == -16 && !v56 )
          v56 = result;
        v22 = v20 & (v55 + v22);
        result = (_QWORD *)(v21 + 16LL * v22);
        v24 = *result;
        if ( *result == a2 )
          goto LABEL_9;
        ++v55;
      }
      if ( v56 )
        result = v56;
    }
LABEL_9:
    *(_DWORD *)(a4 + 16) = v23;
    if ( *result != -8 )
      --*(_DWORD *)(a4 + 20);
    *result = a2;
    result[1] = 0;
  }
LABEL_15:
  result[1] = a3;
  return result;
}
