// Function: sub_356BEB0
// Address: 0x356beb0
//
_QWORD *__fastcall sub_356BEB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // r8d
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r14
  int v13; // r11d
  _QWORD *v14; // rax
  unsigned int v15; // r13d
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // r9
  _QWORD *result; // rax
  int v20; // eax
  int v21; // r14d
  _QWORD *v22; // r9
  unsigned int v23; // r13d
  unsigned int v24; // r10d
  _QWORD *v25; // rax
  __int64 v26; // rcx
  int v27; // eax
  int v28; // eax
  __int64 v29; // r8
  unsigned int v30; // esi
  int v31; // ecx
  __int64 v32; // rdi
  int v33; // edi
  int v34; // ecx
  int v35; // r10d
  int v36; // eax
  int v37; // eax
  int v38; // esi
  __int64 v39; // rdi
  _QWORD *v40; // r8
  unsigned int v41; // r13d
  int v42; // r10d
  __int64 v43; // rax
  int v44; // eax
  int v45; // edx
  __int64 v46; // rdi
  unsigned int v47; // r13d
  __int64 v48; // rsi
  int v49; // r9d
  _QWORD *v50; // r8
  int v51; // eax
  int v52; // edx
  __int64 v53; // rdi
  int v54; // r9d
  unsigned int v55; // r13d
  __int64 v56; // rsi
  int v57; // r11d
  _QWORD *v58; // r10
  __int64 v59; // [rsp+8h] [rbp-28h]
  __int64 v60; // [rsp+8h] [rbp-28h]

  v6 = *(_DWORD *)(a4 + 24);
  v7 = *(_QWORD *)(a4 + 8);
  if ( !v6 )
  {
    ++*(_QWORD *)a4;
    goto LABEL_13;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a3 == *v10 )
  {
LABEL_3:
    if ( v10 != (__int64 *)(v7 + 16LL * v6) )
    {
      v12 = v10[1];
      v13 = 1;
      v14 = 0;
      v15 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      v16 = v15 & v8;
      v17 = (_QWORD *)(v7 + 16LL * (v15 & v8));
      v18 = *v17;
      if ( *v17 == a2 )
      {
LABEL_5:
        result = v17 + 1;
LABEL_6:
        *result = v12;
        return result;
      }
      while ( v18 != -4096 )
      {
        if ( v18 == -8192 && !v14 )
          v14 = v17;
        v16 = v8 & (v13 + v16);
        v17 = (_QWORD *)(v7 + 16LL * v16);
        v18 = *v17;
        if ( *v17 == a2 )
          goto LABEL_5;
        ++v13;
      }
      v33 = *(_DWORD *)(a4 + 16);
      if ( !v14 )
        v14 = v17;
      ++*(_QWORD *)a4;
      v34 = v33 + 1;
      if ( 4 * (v33 + 1) >= 3 * v6 )
      {
        sub_356BCD0(a4, 2 * v6);
        v44 = *(_DWORD *)(a4 + 24);
        if ( v44 )
        {
          v45 = v44 - 1;
          v46 = *(_QWORD *)(a4 + 8);
          v47 = (v44 - 1) & v15;
          v34 = *(_DWORD *)(a4 + 16) + 1;
          v14 = (_QWORD *)(v46 + 16LL * v47);
          v48 = *v14;
          if ( *v14 == a2 )
            goto LABEL_28;
          v49 = 1;
          v50 = 0;
          while ( v48 != -4096 )
          {
            if ( v48 == -8192 && !v50 )
              v50 = v14;
            v47 = v45 & (v49 + v47);
            v14 = (_QWORD *)(v46 + 16LL * v47);
            v48 = *v14;
            if ( *v14 == a2 )
              goto LABEL_28;
            ++v49;
          }
LABEL_53:
          if ( v50 )
            v14 = v50;
          goto LABEL_28;
        }
      }
      else
      {
        if ( v6 - *(_DWORD *)(a4 + 20) - v34 > v6 >> 3 )
        {
LABEL_28:
          *(_DWORD *)(a4 + 16) = v34;
          if ( *v14 != -4096 )
            --*(_DWORD *)(a4 + 20);
          *v14 = a2;
          result = v14 + 1;
          *result = 0;
          goto LABEL_6;
        }
        sub_356BCD0(a4, v6);
        v51 = *(_DWORD *)(a4 + 24);
        if ( v51 )
        {
          v52 = v51 - 1;
          v53 = *(_QWORD *)(a4 + 8);
          v54 = 1;
          v55 = (v51 - 1) & v15;
          v50 = 0;
          v34 = *(_DWORD *)(a4 + 16) + 1;
          v14 = (_QWORD *)(v53 + 16LL * v55);
          v56 = *v14;
          if ( *v14 == a2 )
            goto LABEL_28;
          while ( v56 != -4096 )
          {
            if ( !v50 && v56 == -8192 )
              v50 = v14;
            v55 = v52 & (v54 + v55);
            v14 = (_QWORD *)(v53 + 16LL * v55);
            v56 = *v14;
            if ( *v14 == a2 )
              goto LABEL_28;
            ++v54;
          }
          goto LABEL_53;
        }
      }
LABEL_84:
      ++*(_DWORD *)(a4 + 16);
      BUG();
    }
  }
  else
  {
    v20 = 1;
    while ( v11 != -4096 )
    {
      v35 = v20 + 1;
      v9 = v8 & (v20 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a3 == *v10 )
        goto LABEL_3;
      v20 = v35;
    }
  }
  v21 = 1;
  v22 = 0;
  v23 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v24 = v23 & v8;
  v25 = (_QWORD *)(v7 + 16LL * (v23 & v8));
  v26 = *v25;
  if ( *v25 != a2 )
  {
    while ( v26 != -4096 )
    {
      if ( !v22 && v26 == -8192 )
        v22 = v25;
      v24 = v8 & (v21 + v24);
      v25 = (_QWORD *)(v7 + 16LL * v24);
      v26 = *v25;
      if ( *v25 == a2 )
        goto LABEL_10;
      ++v21;
    }
    if ( !v22 )
      v22 = v25;
    v36 = *(_DWORD *)(a4 + 16);
    ++*(_QWORD *)a4;
    v31 = v36 + 1;
    if ( 4 * (v36 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a4 + 20) - v31 <= v6 >> 3 )
      {
        v60 = a3;
        sub_356BCD0(a4, v6);
        v37 = *(_DWORD *)(a4 + 24);
        if ( !v37 )
          goto LABEL_84;
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a4 + 8);
        v40 = 0;
        v41 = (v37 - 1) & v23;
        v42 = 1;
        v31 = *(_DWORD *)(a4 + 16) + 1;
        a3 = v60;
        v22 = (_QWORD *)(v39 + 16LL * v41);
        v43 = *v22;
        if ( *v22 != a2 )
        {
          while ( v43 != -4096 )
          {
            if ( !v40 && v43 == -8192 )
              v40 = v22;
            v41 = v38 & (v42 + v41);
            v22 = (_QWORD *)(v39 + 16LL * v41);
            v43 = *v22;
            if ( *v22 == a2 )
              goto LABEL_15;
            ++v42;
          }
          if ( v40 )
            v22 = v40;
        }
      }
      goto LABEL_15;
    }
LABEL_13:
    v59 = a3;
    sub_356BCD0(a4, 2 * v6);
    v27 = *(_DWORD *)(a4 + 24);
    if ( !v27 )
      goto LABEL_84;
    v28 = v27 - 1;
    v29 = *(_QWORD *)(a4 + 8);
    v30 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v31 = *(_DWORD *)(a4 + 16) + 1;
    a3 = v59;
    v22 = (_QWORD *)(v29 + 16LL * v30);
    v32 = *v22;
    if ( *v22 != a2 )
    {
      v57 = 1;
      v58 = 0;
      while ( v32 != -4096 )
      {
        if ( v32 == -8192 && !v58 )
          v58 = v22;
        v30 = v28 & (v57 + v30);
        v22 = (_QWORD *)(v29 + 16LL * v30);
        v32 = *v22;
        if ( *v22 == a2 )
          goto LABEL_15;
        ++v57;
      }
      if ( v58 )
        v22 = v58;
    }
LABEL_15:
    *(_DWORD *)(a4 + 16) = v31;
    if ( *v22 != -4096 )
      --*(_DWORD *)(a4 + 20);
    *v22 = a2;
    result = v22 + 1;
    v22[1] = 0;
    goto LABEL_11;
  }
LABEL_10:
  result = v25 + 1;
LABEL_11:
  *result = a3;
  return result;
}
