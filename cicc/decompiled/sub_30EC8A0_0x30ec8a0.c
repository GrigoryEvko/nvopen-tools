// Function: sub_30EC8A0
// Address: 0x30ec8a0
//
_QWORD *__fastcall sub_30EC8A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v5; // rsi
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned int v14; // r8d
  __int64 v15; // r9
  int v16; // r14d
  _QWORD *v17; // rax
  unsigned int v18; // edi
  _QWORD *v19; // rsi
  __int64 v20; // r12
  _QWORD *result; // rax
  unsigned int v22; // r8d
  int v23; // r12d
  _QWORD *v24; // rdx
  unsigned int v25; // edi
  _QWORD *v26; // rax
  __int64 v27; // r10
  int v28; // eax
  int v29; // r11d
  __int64 v30; // r10
  unsigned int v31; // edi
  int v32; // esi
  __int64 v33; // r9
  int v34; // eax
  int v35; // eax
  int v36; // eax
  int v37; // r10d
  __int64 v38; // r9
  _QWORD *v39; // rdi
  unsigned int v40; // ebx
  int v41; // eax
  __int64 v42; // r8
  int v43; // ecx
  int v44; // edi
  int v45; // eax
  int v46; // r10d
  __int64 v47; // rbx
  __int64 v48; // r9
  __int64 v49; // r8
  int v50; // esi
  _QWORD *v51; // r11
  int v52; // eax
  int v53; // r9d
  __int64 v54; // r10
  __int64 v55; // rbx
  int v56; // esi
  __int64 v57; // r8
  int v58; // r10d
  int v59; // eax
  _QWORD *v60; // r8
  int v61; // r11d
  __int64 v62; // rax
  __int64 v63; // [rsp+0h] [rbp-40h]

  v2 = a1 + 8;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 32);
  v63 = v2;
  if ( !v6 )
  {
    v10 = *(_QWORD *)(a2 + 56);
    v11 = a2 + 48;
    if ( a2 + 48 != v10 )
      goto LABEL_6;
    v22 = 0;
    goto LABEL_20;
  }
  v7 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
LABEL_3:
    *v8 = -8192;
    --*(_DWORD *)(a1 + 24);
    ++*(_DWORD *)(a1 + 28);
    v10 = *(_QWORD *)(a2 + 56);
    v11 = a2 + 48;
    if ( v10 != a2 + 48 )
      goto LABEL_6;
LABEL_13:
    v22 = *(_DWORD *)(a1 + 32);
    if ( v22 )
    {
      v5 = *(_QWORD *)(a1 + 16);
      goto LABEL_15;
    }
LABEL_20:
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_21;
  }
  v34 = 1;
  while ( v9 != -4096 )
  {
    v58 = v34 + 1;
    v7 = (v6 - 1) & (v34 + v7);
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
      goto LABEL_3;
    v34 = v58;
  }
  v10 = *(_QWORD *)(a2 + 56);
  v11 = a2 + 48;
  if ( a2 + 48 != v10 )
  {
    while ( 1 )
    {
LABEL_6:
      v12 = v10 - 24;
      if ( !v10 )
        v12 = 0;
      v13 = v12;
      if ( (**(unsigned __int8 (__fastcall ***)(__int64, __int64))a1)(a1, v12) )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v10 == v11 )
        goto LABEL_13;
    }
    v14 = *(_DWORD *)(a1 + 32);
    if ( v14 )
    {
      v15 = *(_QWORD *)(a1 + 16);
      v16 = 1;
      v17 = 0;
      v18 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (_QWORD *)(v15 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == a2 )
      {
LABEL_11:
        result = v19 + 1;
LABEL_12:
        *result = v13;
        return result;
      }
      while ( v20 != -4096 )
      {
        if ( v20 == -8192 && !v17 )
          v17 = v19;
        v18 = (v14 - 1) & (v16 + v18);
        v19 = (_QWORD *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( *v19 == a2 )
          goto LABEL_11;
        ++v16;
      }
      v43 = *(_DWORD *)(a1 + 24);
      if ( !v17 )
        v17 = v19;
      ++*(_QWORD *)(a1 + 8);
      v44 = v43 + 1;
      if ( 4 * (v43 + 1) < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a1 + 28) - v44 > v14 >> 3 )
        {
LABEL_56:
          *(_DWORD *)(a1 + 24) = v44;
          if ( *v17 != -4096 )
            --*(_DWORD *)(a1 + 28);
          v17[1] = 0;
          result = v17 + 1;
          *(result - 1) = a2;
          goto LABEL_12;
        }
        sub_30EC6C0(v63, v14);
        v52 = *(_DWORD *)(a1 + 32);
        if ( v52 )
        {
          v53 = v52 - 1;
          v54 = *(_QWORD *)(a1 + 16);
          v51 = 0;
          LODWORD(v55) = (v52 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v56 = 1;
          v44 = *(_DWORD *)(a1 + 24) + 1;
          v17 = (_QWORD *)(v54 + 16LL * (unsigned int)v55);
          v57 = *v17;
          if ( *v17 == a2 )
            goto LABEL_56;
          while ( v57 != -4096 )
          {
            if ( !v51 && v57 == -8192 )
              v51 = v17;
            v55 = v53 & (unsigned int)(v55 + v56);
            v17 = (_QWORD *)(v54 + 16 * v55);
            v57 = *v17;
            if ( *v17 == a2 )
              goto LABEL_56;
            ++v56;
          }
          goto LABEL_64;
        }
        goto LABEL_97;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    sub_30EC6C0(v63, 2 * v14);
    v45 = *(_DWORD *)(a1 + 32);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a1 + 16);
      v48 = (v45 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = (_QWORD *)(v47 + 16 * v48);
      v44 = *(_DWORD *)(a1 + 24) + 1;
      v49 = *v17;
      if ( *v17 == a2 )
        goto LABEL_56;
      v50 = 1;
      v51 = 0;
      while ( v49 != -4096 )
      {
        if ( v49 == -8192 && !v51 )
          v51 = v17;
        v48 = v46 & (unsigned int)(v48 + v50);
        v17 = (_QWORD *)(v47 + 16 * v48);
        v49 = *v17;
        if ( *v17 == a2 )
          goto LABEL_56;
        ++v50;
      }
LABEL_64:
      if ( v51 )
        v17 = v51;
      goto LABEL_56;
    }
LABEL_97:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
  v22 = v6;
LABEL_15:
  v23 = 1;
  v24 = 0;
  v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (_QWORD *)(v5 + 16LL * v25);
  v27 = *v26;
  if ( *v26 != a2 )
  {
    while ( v27 != -4096 )
    {
      if ( v27 == -8192 && !v24 )
        v24 = v26;
      v25 = (v22 - 1) & (v23 + v25);
      v26 = (_QWORD *)(v5 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == a2 )
        goto LABEL_16;
      ++v23;
    }
    if ( !v24 )
      v24 = v26;
    v35 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v32 = v35 + 1;
    if ( 4 * (v35 + 1) < 3 * v22 )
    {
      if ( v22 - *(_DWORD *)(a1 + 28) - v32 > v22 >> 3 )
      {
LABEL_23:
        *(_DWORD *)(a1 + 24) = v32;
        if ( *v24 != -4096 )
          --*(_DWORD *)(a1 + 28);
        v24[1] = 0;
        *v24 = a2;
        result = v24 + 1;
        goto LABEL_17;
      }
      sub_30EC6C0(v63, v22);
      v36 = *(_DWORD *)(a1 + 32);
      if ( v36 )
      {
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 16);
        v39 = 0;
        v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v32 = *(_DWORD *)(a1 + 24) + 1;
        v41 = 1;
        v24 = (_QWORD *)(v38 + 16LL * v40);
        v42 = *v24;
        if ( *v24 != a2 )
        {
          while ( v42 != -4096 )
          {
            if ( v42 == -8192 && !v39 )
              v39 = v24;
            v61 = v41 + 1;
            v62 = v37 & (v40 + v41);
            v40 = v62;
            v24 = (_QWORD *)(v38 + 16 * v62);
            v42 = *v24;
            if ( *v24 == a2 )
              goto LABEL_23;
            v41 = v61;
          }
          if ( v39 )
            v24 = v39;
        }
        goto LABEL_23;
      }
LABEL_98:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_21:
    sub_30EC6C0(v63, 2 * v22);
    v28 = *(_DWORD *)(a1 + 32);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 16);
      v31 = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = *(_DWORD *)(a1 + 24) + 1;
      v24 = (_QWORD *)(v30 + 16LL * v31);
      v33 = *v24;
      if ( *v24 != a2 )
      {
        v59 = 1;
        v60 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v60 )
            v60 = v24;
          v31 = v29 & (v59 + v31);
          v24 = (_QWORD *)(v30 + 16LL * v31);
          v33 = *v24;
          if ( *v24 == a2 )
            goto LABEL_23;
          ++v59;
        }
        if ( v60 )
          v24 = v60;
      }
      goto LABEL_23;
    }
    goto LABEL_98;
  }
LABEL_16:
  result = v26 + 1;
LABEL_17:
  *result = 0;
  return result;
}
