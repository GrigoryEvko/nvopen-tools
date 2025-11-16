// Function: sub_E65B10
// Address: 0xe65b10
//
__int64 __fastcall sub_E65B10(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // r13
  _QWORD *v9; // rcx
  _QWORD *v10; // rcx
  __int64 (*v11)(); // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r9
  char v15; // al
  __int64 v16; // r8
  char v17; // al
  int v18; // eax
  int v19; // esi
  unsigned int v20; // eax
  __int64 *v21; // rdi
  __int64 v22; // r10
  int v23; // edi
  int v24; // eax
  unsigned int v25; // eax
  __int64 *v26; // rsi
  __int64 *i; // r13
  __int64 v28; // rsi
  __int64 (*v29)(); // rax
  int v30; // eax
  int v31; // esi
  unsigned int v32; // eax
  __int64 v33; // r10
  int v34; // eax
  int v35; // esi
  unsigned int v36; // eax
  __int64 v37; // r10
  int v38; // edi
  int v39; // r11d
  int v40; // eax
  unsigned int v41; // eax
  __int64 *v42; // rdi
  __int64 result; // rax
  __int64 (*v44)(); // rax
  int v45; // eax
  unsigned int v46; // eax
  __int64 v47; // rsi
  int v48; // esi
  int v49; // r10d
  int v50; // edi
  int v51; // r11d
  int v52; // edi
  int v53; // r10d
  int v54; // r11d
  int v55; // edi
  int v56; // edi
  unsigned int v57; // eax
  __int64 v58; // rsi
  unsigned int v59; // r10d
  int v60; // edi
  int v61; // edi
  unsigned int v62; // eax
  __int64 v63; // rsi
  unsigned int v64; // r10d
  unsigned int v65; // r10d
  __int64 v66; // [rsp+0h] [rbp-40h]
  __int64 v67; // [rsp+0h] [rbp-40h]
  __int64 *v68; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD **)(a1 + 1832);
  v5 = 8LL * *(unsigned int *)(a1 + 1840);
  v68 = &v4[(unsigned __int64)v5 / 8];
  v6 = v5 >> 3;
  v7 = v5 >> 5;
  if ( !v7 )
  {
LABEL_46:
    switch ( v6 )
    {
      case 2LL:
        v44 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
        break;
      case 3LL:
        v44 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
        if ( v44 != sub_C14060 )
        {
          if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v44)(a2, *v4) )
          {
            v60 = *(_DWORD *)(a1 + 1824);
            v14 = *(_QWORD *)(a1 + 1808);
            if ( !v60 )
              goto LABEL_22;
            v16 = *v4;
            v61 = v60 - 1;
            v62 = v61 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
            v13 = v14 + 8LL * v62;
            v63 = *(_QWORD *)v13;
            if ( *v4 != *(_QWORD *)v13 )
            {
              v13 = 1;
              while ( v63 != -4096 )
              {
                v64 = v13 + 1;
                v62 = v61 & (v13 + v62);
                v13 = v14 + 8LL * v62;
                v63 = *(_QWORD *)v13;
                if ( v16 == *(_QWORD *)v13 )
                  goto LABEL_57;
                v13 = v64;
              }
              goto LABEL_22;
            }
            goto LABEL_57;
          }
          v44 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
        }
        ++v4;
        break;
      case 1LL:
        v44 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
LABEL_53:
        if ( v44 == sub_C14060 || ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v44)(a2, *v4) )
          goto LABEL_49;
        v45 = *(_DWORD *)(a1 + 1824);
        v16 = *(_QWORD *)(a1 + 1808);
        if ( !v45 )
          goto LABEL_22;
        v14 = (unsigned int)(v45 - 1);
        v46 = v14 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v13 = v16 + 8LL * v46;
        v47 = *(_QWORD *)v13;
        if ( *v4 != *(_QWORD *)v13 )
        {
          v13 = 1;
          while ( v47 != -4096 )
          {
            v65 = v13 + 1;
            v46 = v14 & (v13 + v46);
            v13 = v16 + 8LL * v46;
            v47 = *(_QWORD *)v13;
            if ( *v4 == *(_QWORD *)v13 )
              goto LABEL_57;
            v13 = v65;
          }
          goto LABEL_22;
        }
LABEL_57:
        *(_QWORD *)v13 = -8192;
        --*(_DWORD *)(a1 + 1816);
        ++*(_DWORD *)(a1 + 1820);
        goto LABEL_22;
      default:
LABEL_49:
        v4 = v68;
        goto LABEL_42;
    }
    if ( v44 != sub_C14060 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v44)(a2, *v4) )
      {
        v55 = *(_DWORD *)(a1 + 1824);
        v14 = *(_QWORD *)(a1 + 1808);
        if ( !v55 )
          goto LABEL_22;
        v16 = *v4;
        v56 = v55 - 1;
        v57 = v56 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v13 = v14 + 8LL * v57;
        v58 = *(_QWORD *)v13;
        if ( *v4 != *(_QWORD *)v13 )
        {
          v13 = 1;
          while ( v58 != -4096 )
          {
            v59 = v13 + 1;
            v57 = v56 & (v13 + v57);
            v13 = v14 + 8LL * v57;
            v58 = *(_QWORD *)v13;
            if ( v16 == *(_QWORD *)v13 )
              goto LABEL_57;
            v13 = v59;
          }
          goto LABEL_22;
        }
        goto LABEL_57;
      }
      v44 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
    }
    ++v4;
    goto LABEL_53;
  }
  v8 = &v4[4 * v7];
  while ( 1 )
  {
    v11 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
    if ( v11 == sub_C14060 )
      goto LABEL_3;
    if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v11)(a2, *v4) )
    {
      v24 = *(_DWORD *)(a1 + 1824);
      v16 = *(_QWORD *)(a1 + 1808);
      if ( v24 )
      {
        v13 = (unsigned int)(v24 - 1);
        v25 = v13 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v26 = (__int64 *)(v16 + 8LL * v25);
        v14 = *v26;
        if ( *v4 == *v26 )
        {
LABEL_21:
          *v26 = -8192;
          --*(_DWORD *)(a1 + 1816);
          ++*(_DWORD *)(a1 + 1820);
        }
        else
        {
          v48 = 1;
          while ( v14 != -4096 )
          {
            v49 = v48 + 1;
            v25 = v13 & (v48 + v25);
            v26 = (__int64 *)(v16 + 8LL * v25);
            v14 = *v26;
            if ( *v4 == *v26 )
              goto LABEL_21;
            v48 = v49;
          }
        }
      }
      goto LABEL_22;
    }
    v11 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
    if ( v11 == sub_C14060 )
    {
LABEL_3:
      v9 = v4 + 2;
      if ( v11 != sub_C14060 )
        break;
      goto LABEL_4;
    }
    v15 = ((__int64 (__fastcall *)(__int64, _QWORD))v11)(a2, v4[1]);
    v13 = (__int64)(v4 + 1);
    if ( !v15 )
    {
      v30 = *(_DWORD *)(a1 + 1824);
      v14 = *(_QWORD *)(a1 + 1808);
      if ( !v30 )
        goto LABEL_18;
      v16 = v4[1];
      v31 = v30 - 1;
      v32 = (v30 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v21 = (__int64 *)(v14 + 8LL * v32);
      v33 = *v21;
      if ( *v21 == v16 )
        goto LABEL_32;
      v50 = 1;
      while ( v33 != -4096 )
      {
        v51 = v50 + 1;
        v32 = v31 & (v50 + v32);
        v21 = (__int64 *)(v14 + 8LL * v32);
        v33 = *v21;
        if ( v16 == *v21 )
          goto LABEL_32;
        v50 = v51;
      }
LABEL_18:
      v4 = (_QWORD *)v13;
      goto LABEL_22;
    }
    v9 = v4 + 2;
    v11 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
    if ( v11 != sub_C14060 )
      break;
LABEL_4:
    v10 = v4 + 3;
    if ( v11 != sub_C14060 )
      goto LABEL_13;
LABEL_5:
    v4 += 4;
    if ( v8 == v4 )
    {
      v6 = v68 - v4;
      goto LABEL_46;
    }
  }
  v66 = (__int64)v9;
  v17 = ((__int64 (__fastcall *)(__int64, _QWORD))v11)(a2, v4[2]);
  v13 = v66;
  if ( !v17 )
  {
    v34 = *(_DWORD *)(a1 + 1824);
    v14 = *(_QWORD *)(a1 + 1808);
    if ( !v34 )
      goto LABEL_18;
    v16 = v4[2];
    v35 = v34 - 1;
    v36 = (v34 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v21 = (__int64 *)(v14 + 8LL * v36);
    v37 = *v21;
    if ( v16 == *v21 )
      goto LABEL_32;
    v38 = 1;
    while ( v37 != -4096 )
    {
      v39 = v38 + 1;
      v36 = v35 & (v38 + v36);
      v21 = (__int64 *)(v14 + 8LL * v36);
      v37 = *v21;
      if ( v16 == *v21 )
        goto LABEL_32;
      v38 = v39;
    }
    goto LABEL_18;
  }
  v10 = v4 + 3;
  v11 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
  if ( v11 == sub_C14060 )
    goto LABEL_5;
LABEL_13:
  v67 = (__int64)v10;
  if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v11)(a2, v4[3]) )
    goto LABEL_5;
  v18 = *(_DWORD *)(a1 + 1824);
  v14 = *(_QWORD *)(a1 + 1808);
  v13 = v67;
  if ( !v18 )
    goto LABEL_18;
  v16 = v4[3];
  v19 = v18 - 1;
  v20 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v21 = (__int64 *)(v14 + 8LL * v20);
  v22 = *v21;
  if ( *v21 != v16 )
  {
    v23 = 1;
    while ( v22 != -4096 )
    {
      v54 = v23 + 1;
      v20 = v19 & (v23 + v20);
      v21 = (__int64 *)(v14 + 8LL * v20);
      v22 = *v21;
      if ( v16 == *v21 )
        goto LABEL_32;
      v23 = v54;
    }
    goto LABEL_18;
  }
LABEL_32:
  *v21 = -8192;
  v4 = (_QWORD *)v13;
  --*(_DWORD *)(a1 + 1816);
  ++*(_DWORD *)(a1 + 1820);
LABEL_22:
  if ( v68 != v4 )
  {
    for ( i = v4 + 1; v68 != i; ++*(_DWORD *)(a1 + 1820) )
    {
      while ( 1 )
      {
        v28 = *i;
        v29 = *(__int64 (**)())(*(_QWORD *)a2 + 1272LL);
        if ( v29 != sub_C14060 )
          break;
LABEL_25:
        *v4++ = v28;
LABEL_26:
        if ( v68 == ++i )
          goto LABEL_42;
      }
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))v29)(
             a2,
             v28,
             v12,
             v13,
             v16,
             v14) )
      {
        v28 = *i;
        goto LABEL_25;
      }
      v40 = *(_DWORD *)(a1 + 1824);
      v16 = *(_QWORD *)(a1 + 1808);
      if ( !v40 )
        goto LABEL_26;
      v13 = (unsigned int)(v40 - 1);
      v41 = v13 & (((unsigned int)*i >> 9) ^ ((unsigned int)*i >> 4));
      v42 = (__int64 *)(v16 + 8LL * v41);
      v14 = *v42;
      if ( *v42 != *i )
      {
        v52 = 1;
        while ( v14 != -4096 )
        {
          v53 = v52 + 1;
          v41 = v13 & (v52 + v41);
          v42 = (__int64 *)(v16 + 8LL * v41);
          v14 = *v42;
          if ( *i == *v42 )
            goto LABEL_41;
          v52 = v53;
        }
        goto LABEL_26;
      }
LABEL_41:
      *v42 = -8192;
      ++i;
      --*(_DWORD *)(a1 + 1816);
    }
  }
LABEL_42:
  result = *(_QWORD *)(a1 + 1832);
  if ( v4 != (_QWORD *)(result + 8LL * *(unsigned int *)(a1 + 1840)) )
    *(_DWORD *)(a1 + 1840) = ((__int64)v4 - result) >> 3;
  return result;
}
