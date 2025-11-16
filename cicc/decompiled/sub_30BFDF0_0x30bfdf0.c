// Function: sub_30BFDF0
// Address: 0x30bfdf0
//
__int64 __fastcall sub_30BFDF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 i; // r12
  __int64 v7; // rbx
  __int64 *v8; // r15
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned int v11; // r8d
  __int64 v12; // rdi
  unsigned int v13; // r11d
  _QWORD *v14; // rax
  __int64 v15; // r10
  unsigned __int64 v16; // rcx
  int v17; // r15d
  _QWORD *v18; // rdx
  unsigned int v19; // r11d
  _QWORD *v20; // rax
  __int64 v21; // r10
  __int64 *v22; // rdx
  unsigned int v23; // esi
  __int64 *v24; // r14
  __int64 v25; // r15
  int v26; // esi
  int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // edx
  int v30; // eax
  _QWORD *v31; // rcx
  __int64 v32; // rdi
  int v33; // r10d
  _QWORD *v34; // r11
  int v35; // eax
  int v36; // eax
  int v38; // eax
  int v39; // eax
  int v40; // ecx
  __int64 v41; // r8
  unsigned int v42; // edi
  __int64 v43; // rsi
  int v44; // r11d
  _QWORD *v45; // r10
  int v46; // esi
  int v47; // esi
  __int64 v48; // r8
  int v49; // r10d
  unsigned int v50; // edx
  __int64 v51; // rdi
  int v52; // eax
  int v53; // ecx
  __int64 v54; // r8
  int v55; // r11d
  unsigned int v56; // edi
  __int64 v57; // rsi
  int v58; // [rsp+8h] [rbp-68h]
  __int64 v59; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+10h] [rbp-60h]
  unsigned int v62; // [rsp+10h] [rbp-60h]
  __int64 v63; // [rsp+18h] [rbp-58h]
  __int64 v64; // [rsp+18h] [rbp-58h]
  __int64 v65; // [rsp+20h] [rbp-50h]
  unsigned int v66; // [rsp+20h] [rbp-50h]
  __int64 v68; // [rsp+28h] [rbp-48h]
  __int64 v69; // [rsp+28h] [rbp-48h]
  __int64 v71[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = a1;
  v71[0] = a5;
  v63 = a3 & 1;
  if ( a2 >= (a3 - 1) / 2 )
  {
    if ( (a3 & 1) != 0 )
    {
      v24 = (__int64 *)(a1 + 8 * a2);
      goto LABEL_41;
    }
    v7 = a2;
  }
  else
  {
    v65 = (a3 - 1) / 2;
    for ( i = a2; ; i = v7 )
    {
      v7 = 2 * (i + 1);
      v8 = (__int64 *)(a1 + 16 * (i + 1));
      if ( sub_30BBEA0(v71, *v8, *(v8 - 1)) )
      {
        --v7;
        v8 = (__int64 *)(a1 + 8 * v7);
      }
      *(_QWORD *)(a1 + 8 * i) = *v8;
      if ( v7 >= v65 )
        break;
    }
    v5 = a1;
    if ( v63 )
      goto LABEL_8;
  }
  if ( (a3 - 2) / 2 == v7 )
  {
    *(_QWORD *)(v5 + 8 * v7) = *(_QWORD *)(v5 + 8 * (2 * v7 + 2) - 8);
    v7 = 2 * v7 + 1;
  }
LABEL_8:
  v9 = v71[0];
  if ( v7 <= a2 )
  {
    v24 = (__int64 *)(v5 + 8 * v7);
    goto LABEL_41;
  }
  v10 = (v7 - 1) / 2;
  v66 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  v64 = v71[0] + 96;
  while ( 1 )
  {
    v23 = *(_DWORD *)(v9 + 120);
    v24 = (__int64 *)(v5 + 8 * v10);
    v25 = *v24;
    if ( v23 )
    {
      v11 = v23 - 1;
      v12 = *(_QWORD *)(v9 + 104);
      v13 = (v23 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v14 = (_QWORD *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v25 == *v14 )
      {
LABEL_11:
        v16 = v14[1];
        goto LABEL_12;
      }
      v58 = 1;
      v31 = 0;
      while ( v15 != -4096 )
      {
        if ( !v31 && v15 == -8192 )
          v31 = v14;
        v13 = v11 & (v58 + v13);
        v14 = (_QWORD *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v25 == *v14 )
          goto LABEL_11;
        ++v58;
      }
      if ( !v31 )
        v31 = v14;
      v38 = *(_DWORD *)(v9 + 112);
      ++*(_QWORD *)(v9 + 96);
      v30 = v38 + 1;
      if ( 4 * v30 < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(v9 + 116) - v30 > v23 >> 3 )
          goto LABEL_48;
        v59 = v5;
        v62 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
        sub_30BBCC0(v64, v23);
        v46 = *(_DWORD *)(v9 + 120);
        if ( !v46 )
          goto LABEL_95;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(v9 + 104);
        v34 = 0;
        v5 = v59;
        v49 = 1;
        v50 = v47 & v62;
        v30 = *(_DWORD *)(v9 + 112) + 1;
        v31 = (_QWORD *)(v48 + 16LL * (v47 & v62));
        v51 = *v31;
        if ( v25 == *v31 )
          goto LABEL_48;
        while ( v51 != -4096 )
        {
          if ( !v34 && v51 == -8192 )
            v34 = v31;
          v50 = v47 & (v49 + v50);
          v31 = (_QWORD *)(v48 + 16LL * v50);
          v51 = *v31;
          if ( v25 == *v31 )
            goto LABEL_48;
          ++v49;
        }
        goto LABEL_22;
      }
    }
    else
    {
      ++*(_QWORD *)(v9 + 96);
    }
    v61 = v5;
    sub_30BBCC0(v64, 2 * v23);
    v26 = *(_DWORD *)(v9 + 120);
    if ( !v26 )
      goto LABEL_95;
    v27 = v26 - 1;
    v28 = *(_QWORD *)(v9 + 104);
    v5 = v61;
    v29 = v27 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v30 = *(_DWORD *)(v9 + 112) + 1;
    v31 = (_QWORD *)(v28 + 16LL * v29);
    v32 = *v31;
    if ( v25 == *v31 )
      goto LABEL_48;
    v33 = 1;
    v34 = 0;
    while ( v32 != -4096 )
    {
      if ( v32 == -8192 && !v34 )
        v34 = v31;
      v29 = v27 & (v33 + v29);
      v31 = (_QWORD *)(v28 + 16LL * v29);
      v32 = *v31;
      if ( v25 == *v31 )
        goto LABEL_48;
      ++v33;
    }
LABEL_22:
    if ( v34 )
      v31 = v34;
LABEL_48:
    *(_DWORD *)(v9 + 112) = v30;
    if ( *v31 != -4096 )
      --*(_DWORD *)(v9 + 116);
    *v31 = v25;
    v31[1] = 0;
    v23 = *(_DWORD *)(v9 + 120);
    if ( !v23 )
    {
      ++*(_QWORD *)(v9 + 96);
      goto LABEL_52;
    }
    v12 = *(_QWORD *)(v9 + 104);
    v11 = v23 - 1;
    v16 = 0;
LABEL_12:
    v17 = 1;
    v18 = 0;
    v19 = v11 & v66;
    v20 = (_QWORD *)(v12 + 16LL * (v11 & v66));
    v21 = *v20;
    if ( a4 != *v20 )
      break;
LABEL_13:
    v22 = (__int64 *)(v5 + 8 * v7);
    if ( v16 >= v20[1] )
    {
      v24 = (__int64 *)(v5 + 8 * v7);
      goto LABEL_41;
    }
    v7 = v10;
    *v22 = *v24;
    if ( a2 >= v10 )
      goto LABEL_41;
    v10 = (v10 - 1) / 2;
  }
  while ( v21 != -4096 )
  {
    if ( !v18 && v21 == -8192 )
      v18 = v20;
    v19 = v11 & (v17 + v19);
    v20 = (_QWORD *)(v12 + 16LL * v19);
    v21 = *v20;
    if ( a4 == *v20 )
      goto LABEL_13;
    ++v17;
  }
  if ( !v18 )
    v18 = v20;
  v35 = *(_DWORD *)(v9 + 112);
  ++*(_QWORD *)(v9 + 96);
  v36 = v35 + 1;
  if ( 4 * v36 < 3 * v23 )
  {
    if ( v23 - (v36 + *(_DWORD *)(v9 + 116)) > v23 >> 3 )
      goto LABEL_38;
    v69 = v5;
    sub_30BBCC0(v64, v23);
    v52 = *(_DWORD *)(v9 + 120);
    if ( v52 )
    {
      v53 = v52 - 1;
      v54 = *(_QWORD *)(v9 + 104);
      v45 = 0;
      v5 = v69;
      v55 = 1;
      v56 = v53 & v66;
      v36 = *(_DWORD *)(v9 + 112) + 1;
      v18 = (_QWORD *)(v54 + 16LL * (v53 & v66));
      v57 = *v18;
      if ( a4 != *v18 )
      {
        while ( v57 != -4096 )
        {
          if ( !v45 && v57 == -8192 )
            v45 = v18;
          v56 = v53 & (v55 + v56);
          v18 = (_QWORD *)(v54 + 16LL * v56);
          v57 = *v18;
          if ( a4 == *v18 )
            goto LABEL_38;
          ++v55;
        }
LABEL_56:
        if ( v45 )
          v18 = v45;
      }
      goto LABEL_38;
    }
LABEL_95:
    ++*(_DWORD *)(v9 + 112);
    BUG();
  }
LABEL_52:
  v68 = v5;
  sub_30BBCC0(v64, 2 * v23);
  v39 = *(_DWORD *)(v9 + 120);
  if ( !v39 )
    goto LABEL_95;
  v40 = v39 - 1;
  v41 = *(_QWORD *)(v9 + 104);
  v5 = v68;
  v42 = v40 & v66;
  v36 = *(_DWORD *)(v9 + 112) + 1;
  v18 = (_QWORD *)(v41 + 16LL * (v40 & v66));
  v43 = *v18;
  if ( a4 != *v18 )
  {
    v44 = 1;
    v45 = 0;
    while ( v43 != -4096 )
    {
      if ( v43 == -8192 && !v45 )
        v45 = v18;
      v42 = v40 & (v44 + v42);
      v18 = (_QWORD *)(v41 + 16LL * v42);
      v43 = *v18;
      if ( a4 == *v18 )
        goto LABEL_38;
      ++v44;
    }
    goto LABEL_56;
  }
LABEL_38:
  *(_DWORD *)(v9 + 112) = v36;
  if ( *v18 != -4096 )
    --*(_DWORD *)(v9 + 116);
  v18[1] = 0;
  v24 = (__int64 *)(v5 + 8 * v7);
  *v18 = a4;
LABEL_41:
  *v24 = a4;
  return a4;
}
