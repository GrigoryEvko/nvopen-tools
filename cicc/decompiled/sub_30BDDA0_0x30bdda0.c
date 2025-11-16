// Function: sub_30BDDA0
// Address: 0x30bdda0
//
unsigned __int64 __fastcall sub_30BDDA0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v4; // r15
  unsigned int v5; // r10d
  unsigned int v6; // edx
  __int64 v7; // rcx
  int v8; // r14d
  unsigned int v9; // r9d
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // r8
  unsigned __int64 v13; // rdi
  unsigned int v14; // r14d
  unsigned int v15; // r9d
  unsigned __int64 result; // rax
  __int64 v17; // r8
  unsigned int v18; // esi
  __int64 v19; // r13
  _QWORD *v20; // r11
  int v21; // edx
  int v22; // edx
  __int64 v23; // r8
  unsigned int v24; // esi
  int v25; // eax
  __int64 v26; // rcx
  int v27; // r14d
  _QWORD *v28; // r9
  int v29; // eax
  int v30; // eax
  int v31; // esi
  __int64 v32; // rdi
  unsigned int v33; // ecx
  _QWORD *v34; // rdx
  __int64 v35; // r8
  int v36; // r10d
  _QWORD *v37; // r9
  int v38; // eax
  int v39; // edx
  int v40; // edx
  __int64 v41; // r8
  int v42; // r14d
  unsigned int v43; // ecx
  __int64 v44; // rsi
  int v45; // eax
  int v46; // ecx
  __int64 v47; // rdi
  _QWORD *v48; // r8
  unsigned int v49; // r14d
  int v50; // r9d
  __int64 v51; // rsi
  unsigned int v52; // [rsp+4h] [rbp-4Ch]
  int v53; // [rsp+8h] [rbp-48h]
  unsigned int v54; // [rsp+10h] [rbp-40h]
  _QWORD *v55; // [rsp+10h] [rbp-40h]
  _QWORD *v56; // [rsp+10h] [rbp-40h]
  unsigned int v57; // [rsp+10h] [rbp-40h]
  _QWORD *v58; // [rsp+10h] [rbp-40h]
  __int64 v59; // [rsp+18h] [rbp-38h]

  v2 = a1;
  v4 = *a1;
  v59 = a2 + 96;
  v5 = ((unsigned int)*a1 >> 9) ^ ((unsigned int)*a1 >> 4);
  while ( 1 )
  {
    v18 = *(_DWORD *)(a2 + 120);
    v19 = *(v2 - 1);
    v20 = v2;
    if ( v18 )
    {
      v6 = v18 - 1;
      v7 = *(_QWORD *)(a2 + 104);
      v8 = 1;
      v9 = (v18 - 1) & v5;
      v10 = 0;
      v11 = (_QWORD *)(v7 + 16LL * v9);
      v12 = *v11;
      if ( v4 == *v11 )
      {
LABEL_3:
        v13 = v11[1];
        goto LABEL_4;
      }
      while ( v12 != -4096 )
      {
        if ( !v10 && v12 == -8192 )
          v10 = v11;
        v9 = v6 & (v8 + v9);
        v11 = (_QWORD *)(v7 + 16LL * v9);
        v12 = *v11;
        if ( v4 == *v11 )
          goto LABEL_3;
        ++v8;
      }
      if ( !v10 )
        v10 = v11;
      v29 = *(_DWORD *)(a2 + 112);
      ++*(_QWORD *)(a2 + 96);
      v25 = v29 + 1;
      if ( 4 * v25 < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a2 + 116) - v25 > v18 >> 3 )
          goto LABEL_26;
        v57 = v5;
        sub_30BBCC0(v59, v18);
        v39 = *(_DWORD *)(a2 + 120);
        if ( !v39 )
          goto LABEL_83;
        v5 = v57;
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a2 + 104);
        v28 = 0;
        v20 = v2;
        v42 = 1;
        v43 = v40 & v57;
        v25 = *(_DWORD *)(a2 + 112) + 1;
        v10 = (_QWORD *)(v41 + 16LL * (v40 & v57));
        v44 = *v10;
        if ( v4 == *v10 )
          goto LABEL_26;
        while ( v44 != -4096 )
        {
          if ( !v28 && v44 == -8192 )
            v28 = v10;
          v43 = v40 & (v42 + v43);
          v10 = (_QWORD *)(v41 + 16LL * v43);
          v44 = *v10;
          if ( v4 == *v10 )
            goto LABEL_26;
          ++v42;
        }
        goto LABEL_13;
      }
    }
    else
    {
      ++*(_QWORD *)(a2 + 96);
    }
    v54 = v5;
    sub_30BBCC0(v59, 2 * v18);
    v21 = *(_DWORD *)(a2 + 120);
    if ( !v21 )
      goto LABEL_83;
    v5 = v54;
    v22 = v21 - 1;
    v23 = *(_QWORD *)(a2 + 104);
    v20 = v2;
    v24 = v22 & v54;
    v25 = *(_DWORD *)(a2 + 112) + 1;
    v10 = (_QWORD *)(v23 + 16LL * (v22 & v54));
    v26 = *v10;
    if ( v4 == *v10 )
      goto LABEL_26;
    v27 = 1;
    v28 = 0;
    while ( v26 != -4096 )
    {
      if ( !v28 && v26 == -8192 )
        v28 = v10;
      v24 = v22 & (v27 + v24);
      v10 = (_QWORD *)(v23 + 16LL * v24);
      v26 = *v10;
      if ( v4 == *v10 )
        goto LABEL_26;
      ++v27;
    }
LABEL_13:
    if ( v28 )
      v10 = v28;
LABEL_26:
    *(_DWORD *)(a2 + 112) = v25;
    if ( *v10 != -4096 )
      --*(_DWORD *)(a2 + 116);
    *v10 = v4;
    v10[1] = 0;
    v18 = *(_DWORD *)(a2 + 120);
    if ( !v18 )
    {
      ++*(_QWORD *)(a2 + 96);
      goto LABEL_30;
    }
    v7 = *(_QWORD *)(a2 + 104);
    v6 = v18 - 1;
    v13 = 0;
LABEL_4:
    v14 = ((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4);
    v15 = v14 & v6;
    result = v7 + 16LL * (v14 & v6);
    v17 = *(_QWORD *)result;
    if ( v19 != *(_QWORD *)result )
      break;
LABEL_5:
    --v2;
    if ( v13 >= *(_QWORD *)(result + 8) )
      goto LABEL_46;
    v2[1] = *v2;
  }
  v53 = 1;
  v56 = 0;
  v52 = v6;
  while ( 1 )
  {
    v34 = v56;
    if ( v17 == -4096 )
      break;
    if ( !v56 )
    {
      if ( v17 != -8192 )
        result = 0;
      v56 = (_QWORD *)result;
    }
    v15 = v52 & (v53 + v15);
    result = v7 + 16LL * v15;
    v17 = *(_QWORD *)result;
    if ( v19 == *(_QWORD *)result )
      goto LABEL_5;
    ++v53;
  }
  if ( !v56 )
    v34 = (_QWORD *)result;
  v38 = *(_DWORD *)(a2 + 112);
  ++*(_QWORD *)(a2 + 96);
  result = (unsigned int)(v38 + 1);
  if ( 4 * (int)result >= 3 * v18 )
  {
LABEL_30:
    v55 = v20;
    sub_30BBCC0(v59, 2 * v18);
    v30 = *(_DWORD *)(a2 + 120);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a2 + 104);
      v20 = v55;
      v33 = (v30 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      result = (unsigned int)(*(_DWORD *)(a2 + 112) + 1);
      v34 = (_QWORD *)(v32 + 16LL * v33);
      v35 = *v34;
      if ( v19 != *v34 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( !v37 && v35 == -8192 )
            v37 = v34;
          v33 = v31 & (v36 + v33);
          v34 = (_QWORD *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v19 == *v34 )
            goto LABEL_43;
          ++v36;
        }
        if ( v37 )
          v34 = v37;
      }
      goto LABEL_43;
    }
LABEL_83:
    ++*(_DWORD *)(a2 + 112);
    BUG();
  }
  if ( v18 - ((_DWORD)result + *(_DWORD *)(a2 + 116)) <= v18 >> 3 )
  {
    v58 = v20;
    sub_30BBCC0(v59, v18);
    v45 = *(_DWORD *)(a2 + 120);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a2 + 104);
      v48 = 0;
      v49 = (v45 - 1) & v14;
      v20 = v58;
      v50 = 1;
      result = (unsigned int)(*(_DWORD *)(a2 + 112) + 1);
      v34 = (_QWORD *)(v47 + 16LL * v49);
      v51 = *v34;
      if ( v19 != *v34 )
      {
        while ( v51 != -4096 )
        {
          if ( !v48 && v51 == -8192 )
            v48 = v34;
          v49 = v46 & (v50 + v49);
          v34 = (_QWORD *)(v47 + 16LL * v49);
          v51 = *v34;
          if ( v19 == *v34 )
            goto LABEL_43;
          ++v50;
        }
        if ( v48 )
          v34 = v48;
      }
      goto LABEL_43;
    }
    goto LABEL_83;
  }
LABEL_43:
  *(_DWORD *)(a2 + 112) = result;
  if ( *v34 != -4096 )
    --*(_DWORD *)(a2 + 116);
  *v34 = v19;
  v34[1] = 0;
LABEL_46:
  *v20 = v4;
  return result;
}
