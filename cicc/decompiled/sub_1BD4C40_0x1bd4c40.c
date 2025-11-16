// Function: sub_1BD4C40
// Address: 0x1bd4c40
//
__int64 __fastcall sub_1BD4C40(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // r14
  _QWORD *v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r9
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // r12
  __int64 v14; // rdx
  char v15; // di
  __int64 *v16; // r9
  int v17; // esi
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r10
  unsigned int v22; // esi
  unsigned int v23; // ecx
  int v24; // r8d
  unsigned int v25; // r9d
  __int64 v26; // rax
  int v27; // r13d
  __int64 *v28; // r11
  int v29; // r11d
  __int64 *v30; // r10
  int v31; // edi
  int v32; // ecx
  __int64 *v33; // rdx
  int v34; // edx
  int v35; // edx
  __int64 v36; // r8
  unsigned int v37; // esi
  __int64 v38; // rdi
  int v39; // r10d
  __int64 *v40; // r9
  __int64 *v41; // r9
  int v42; // esi
  unsigned int v43; // ecx
  __int64 v44; // r8
  __int64 *v45; // r9
  int v46; // esi
  unsigned int v47; // ecx
  __int64 v48; // rdi
  int v49; // r11d
  __int64 *v50; // r10
  int v51; // edx
  int v52; // esi
  __int64 v53; // rdi
  __int64 *v54; // r8
  unsigned int v55; // r15d
  int v56; // r9d
  __int64 v57; // rdx
  int v58; // esi
  int v59; // esi
  int v60; // r11d
  _QWORD *v61; // [rsp+8h] [rbp-38h]
  _QWORD *v62; // [rsp+8h] [rbp-38h]
  _QWORD *v63; // [rsp+8h] [rbp-38h]
  _QWORD *v64; // [rsp+8h] [rbp-38h]

  v3 = (_QWORD *)sub_1BC27C0(*(_QWORD *)a1);
  *v3 = a2;
  v4 = v3;
  v5 = *(_DWORD *)(*(_QWORD *)a1 + 224LL);
  v6 = **(_QWORD **)(a1 + 8);
  v4[1] = v4;
  v4[2] = 0;
  v4[3] = 0;
  *((_DWORD *)v4 + 20) = v5;
  v4[11] = -1;
  *((_DWORD *)v4 + 24) = -1;
  *((_BYTE *)v4 + 100) = 0;
  *((_DWORD *)v4 + 10) = 0;
  v4[13] = v6;
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD **)(a1 + 8);
  v9 = *(_DWORD *)(*(_QWORD *)a1 + 96LL);
  v10 = *(_QWORD *)a1 + 72LL;
  if ( !v9 )
  {
    ++*(_QWORD *)(v7 + 72);
    goto LABEL_36;
  }
  v11 = *(_QWORD *)(v7 + 80);
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v11 + 88LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
    goto LABEL_3;
  v29 = 1;
  v30 = 0;
  while ( v14 != -8 )
  {
    if ( !v30 && v14 == -16 )
      v30 = v13;
    v12 = (v9 - 1) & (v29 + v12);
    v13 = (__int64 *)(v11 + 88LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
      goto LABEL_3;
    ++v29;
  }
  v31 = *(_DWORD *)(v7 + 88);
  if ( v30 )
    v13 = v30;
  ++*(_QWORD *)(v7 + 72);
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v9 )
  {
LABEL_36:
    v61 = v8;
    sub_1BD4A10(v10, 2 * v9);
    v34 = *(_DWORD *)(v7 + 96);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v7 + 80);
      v37 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v36 + 88LL * v37);
      v32 = *(_DWORD *)(v7 + 88) + 1;
      v8 = v61;
      v38 = *v13;
      if ( a2 != *v13 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( !v40 && v38 == -16 )
            v40 = v13;
          v37 = v35 & (v39 + v37);
          v13 = (__int64 *)(v36 + 88LL * v37);
          v38 = *v13;
          if ( a2 == *v13 )
            goto LABEL_28;
          ++v39;
        }
        if ( v40 )
          v13 = v40;
      }
      goto LABEL_28;
    }
    goto LABEL_97;
  }
  if ( v9 - *(_DWORD *)(v7 + 92) - v32 <= v9 >> 3 )
  {
    v64 = v8;
    sub_1BD4A10(v10, v9);
    v51 = *(_DWORD *)(v7 + 96);
    if ( v51 )
    {
      v52 = v51 - 1;
      v53 = *(_QWORD *)(v7 + 80);
      v54 = 0;
      v55 = (v51 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v56 = 1;
      v32 = *(_DWORD *)(v7 + 88) + 1;
      v8 = v64;
      v13 = (__int64 *)(v53 + 88LL * v55);
      v57 = *v13;
      if ( a2 != *v13 )
      {
        while ( v57 != -8 )
        {
          if ( v57 == -16 && !v54 )
            v54 = v13;
          v55 = v52 & (v56 + v55);
          v13 = (__int64 *)(v53 + 88LL * v55);
          v57 = *v13;
          if ( a2 == *v13 )
            goto LABEL_28;
          ++v56;
        }
        if ( v54 )
          v13 = v54;
      }
      goto LABEL_28;
    }
LABEL_97:
    ++*(_DWORD *)(v7 + 88);
    BUG();
  }
LABEL_28:
  *(_DWORD *)(v7 + 88) = v32;
  if ( *v13 != -8 )
    --*(_DWORD *)(v7 + 92);
  *v13 = a2;
  v33 = v13 + 3;
  v13[1] = 0;
  v13[2] = 1;
  do
  {
    if ( v33 )
      *v33 = -8;
    v33 += 2;
  }
  while ( v33 != v13 + 11 );
LABEL_3:
  v15 = v13[2] & 1;
  if ( v15 )
  {
    v16 = v13 + 3;
    v17 = 3;
  }
  else
  {
    v22 = *((_DWORD *)v13 + 8);
    v16 = (__int64 *)v13[3];
    if ( !v22 )
    {
      v23 = *((_DWORD *)v13 + 4);
      ++v13[1];
      v19 = 0;
      v24 = (v23 >> 1) + 1;
LABEL_10:
      v25 = 3 * v22;
      goto LABEL_11;
    }
    v17 = v22 - 1;
  }
  v18 = v17 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
  v19 = &v16[2 * v18];
  v20 = *v19;
  if ( *v8 == *v19 )
    goto LABEL_6;
  v27 = 1;
  v28 = 0;
  while ( v20 != -8 )
  {
    if ( !v28 && v20 == -16 )
      v28 = v19;
    v18 = v17 & (v27 + v18);
    v19 = &v16[2 * v18];
    v20 = *v19;
    if ( *v8 == *v19 )
      goto LABEL_6;
    ++v27;
  }
  v23 = *((_DWORD *)v13 + 4);
  v25 = 12;
  v22 = 4;
  if ( v28 )
    v19 = v28;
  ++v13[1];
  v24 = (v23 >> 1) + 1;
  if ( !v15 )
  {
    v22 = *((_DWORD *)v13 + 8);
    goto LABEL_10;
  }
LABEL_11:
  if ( 4 * v24 >= v25 )
  {
    v62 = v8;
    sub_1BCABB0((__int64)(v13 + 1), 2 * v22);
    v8 = v62;
    if ( (v13[2] & 1) != 0 )
    {
      v41 = v13 + 3;
      v42 = 3;
    }
    else
    {
      v58 = *((_DWORD *)v13 + 8);
      v41 = (__int64 *)v13[3];
      if ( !v58 )
        goto LABEL_96;
      v42 = v58 - 1;
    }
    v43 = v42 & (((unsigned int)*v62 >> 9) ^ ((unsigned int)*v62 >> 4));
    v19 = &v41[2 * v43];
    v44 = *v19;
    if ( *v19 != *v62 )
    {
      v60 = 1;
      v50 = 0;
      while ( v44 != -8 )
      {
        if ( !v50 && v44 == -16 )
          v50 = v19;
        v43 = v42 & (v60 + v43);
        v19 = &v41[2 * v43];
        v44 = *v19;
        if ( *v62 == *v19 )
          goto LABEL_46;
        ++v60;
      }
      goto LABEL_52;
    }
LABEL_46:
    v23 = *((_DWORD *)v13 + 4);
    goto LABEL_13;
  }
  if ( v22 - *((_DWORD *)v13 + 5) - v24 <= v22 >> 3 )
  {
    v63 = v8;
    sub_1BCABB0((__int64)(v13 + 1), v22);
    v8 = v63;
    if ( (v13[2] & 1) != 0 )
    {
      v45 = v13 + 3;
      v46 = 3;
      goto LABEL_49;
    }
    v59 = *((_DWORD *)v13 + 8);
    v45 = (__int64 *)v13[3];
    if ( v59 )
    {
      v46 = v59 - 1;
LABEL_49:
      v47 = v46 & (((unsigned int)*v63 >> 9) ^ ((unsigned int)*v63 >> 4));
      v19 = &v45[2 * v47];
      v48 = *v19;
      if ( *v19 != *v63 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -8 )
        {
          if ( !v50 && v48 == -16 )
            v50 = v19;
          v47 = v46 & (v49 + v47);
          v19 = &v45[2 * v47];
          v48 = *v19;
          if ( *v63 == *v19 )
            goto LABEL_46;
          ++v49;
        }
LABEL_52:
        if ( v50 )
          v19 = v50;
        goto LABEL_46;
      }
      goto LABEL_46;
    }
LABEL_96:
    *((_DWORD *)v13 + 4) = (2 * (*((_DWORD *)v13 + 4) >> 1) + 2) | v13[2] & 1;
    BUG();
  }
LABEL_13:
  *((_DWORD *)v13 + 4) = (2 * (v23 >> 1) + 2) | v23 & 1;
  if ( *v19 != -8 )
    --*((_DWORD *)v13 + 5);
  v26 = *v8;
  v19[1] = 0;
  *v19 = v26;
LABEL_6:
  v19[1] = (__int64)v4;
  return 1;
}
