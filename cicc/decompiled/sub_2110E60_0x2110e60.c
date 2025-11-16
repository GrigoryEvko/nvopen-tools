// Function: sub_2110E60
// Address: 0x2110e60
//
_QWORD *__fastcall sub_2110E60(_QWORD *a1, __int64 a2, unsigned int *a3)
{
  __int64 v6; // r15
  _QWORD *v7; // rcx
  unsigned int v8; // esi
  __int64 v9; // r10
  int v10; // r9d
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 *v13; // r13
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  _DWORD *v17; // rax
  int v18; // r15d
  __int64 v19; // r13
  unsigned int v20; // esi
  __int64 v21; // r10
  __int64 v22; // r8
  unsigned int v23; // edi
  __int64 *v24; // rax
  __int64 v25; // rcx
  _QWORD *v26; // rsi
  unsigned int v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rcx
  _QWORD *result; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  _QWORD *v34; // rcx
  __int64 *v35; // r11
  int v36; // eax
  int v37; // eax
  __int64 *v38; // r11
  int v39; // edi
  int v40; // edi
  int v41; // esi
  int v42; // esi
  __int64 v43; // r9
  unsigned int v44; // edx
  __int64 v45; // r8
  int v46; // r11d
  __int64 *v47; // r10
  int v48; // eax
  int v49; // esi
  __int64 v50; // rcx
  unsigned int v51; // edx
  __int64 v52; // r8
  int v53; // r10d
  __int64 *v54; // r9
  int v55; // esi
  int v56; // esi
  int v57; // r11d
  __int64 v58; // r9
  unsigned int v59; // edx
  __int64 v60; // r8
  int v61; // eax
  int v62; // ecx
  __int64 v63; // r8
  int v64; // r10d
  unsigned int v65; // edx
  __int64 v66; // rsi
  int v67; // [rsp+8h] [rbp-48h]
  int v68; // [rsp+8h] [rbp-48h]
  _QWORD *v69; // [rsp+8h] [rbp-48h]
  _QWORD *v70; // [rsp+8h] [rbp-48h]
  _BYTE v71[12]; // [rsp+14h] [rbp-3Ch]

  v6 = *a1;
  v7 = (_QWORD *)a1[1];
  v8 = *(_DWORD *)(*a1 + 504LL);
  v9 = *a1 + 480LL;
  if ( !v8 )
  {
    ++*(_QWORD *)(v6 + 480);
    goto LABEL_33;
  }
  v10 = v8 - 1;
  v11 = *(_QWORD *)(v6 + 488);
  v12 = (v8 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
  v13 = (__int64 *)(v11 + 72LL * v12);
  v14 = *v13;
  if ( *v7 == *v13 )
  {
LABEL_3:
    v15 = *((unsigned int *)v13 + 4);
    *(_DWORD *)v71 = *(_DWORD *)a1[2];
    *(_QWORD *)&v71[4] = *(_QWORD *)a3;
    if ( (unsigned int)v15 >= *((_DWORD *)v13 + 5) )
    {
      sub_16CD150((__int64)(v13 + 1), v13 + 3, 0, 12, v11, v10);
      v15 = *((unsigned int *)v13 + 4);
    }
    goto LABEL_5;
  }
  v67 = 1;
  v35 = 0;
  while ( v14 != -8 )
  {
    if ( v35 || v14 != -16 )
      v13 = v35;
    v12 = v10 & (v67 + v12);
    v14 = *(_QWORD *)(v11 + 72LL * v12);
    if ( *v7 == v14 )
    {
      v13 = (__int64 *)(v11 + 72LL * v12);
      goto LABEL_3;
    }
    ++v67;
    v35 = v13;
    v13 = (__int64 *)(v11 + 72LL * v12);
  }
  v36 = *(_DWORD *)(v6 + 496);
  if ( v35 )
    v13 = v35;
  ++*(_QWORD *)(v6 + 480);
  v37 = v36 + 1;
  if ( 4 * v37 >= 3 * v8 )
  {
LABEL_33:
    v69 = v7;
    sub_210FEC0(v9, 2 * v8);
    v41 = *(_DWORD *)(v6 + 504);
    if ( v41 )
    {
      v7 = v69;
      v42 = v41 - 1;
      v43 = *(_QWORD *)(v6 + 488);
      v44 = v42 & (((unsigned int)*v69 >> 9) ^ ((unsigned int)*v69 >> 4));
      v13 = (__int64 *)(v43 + 72LL * v44);
      v45 = *v13;
      v37 = *(_DWORD *)(v6 + 496) + 1;
      if ( *v69 == *v13 )
        goto LABEL_20;
      v46 = 1;
      v47 = 0;
      while ( v45 != -8 )
      {
        if ( !v47 && v45 == -16 )
          v47 = v13;
        v44 = v42 & (v46 + v44);
        v13 = (__int64 *)(v43 + 72LL * v44);
        v45 = *v13;
        if ( *v69 == *v13 )
          goto LABEL_20;
        ++v46;
      }
LABEL_37:
      if ( v47 )
        v13 = v47;
      goto LABEL_20;
    }
LABEL_85:
    ++*(_DWORD *)(v6 + 496);
    BUG();
  }
  if ( v8 - *(_DWORD *)(v6 + 500) - v37 <= v8 >> 3 )
  {
    v70 = v7;
    sub_210FEC0(v9, v8);
    v55 = *(_DWORD *)(v6 + 504);
    if ( v55 )
    {
      v7 = v70;
      v56 = v55 - 1;
      v57 = 1;
      v47 = 0;
      v58 = *(_QWORD *)(v6 + 488);
      v59 = v56 & (((unsigned int)*v70 >> 9) ^ ((unsigned int)*v70 >> 4));
      v13 = (__int64 *)(v58 + 72LL * v59);
      v60 = *v13;
      v37 = *(_DWORD *)(v6 + 496) + 1;
      if ( *v70 == *v13 )
        goto LABEL_20;
      while ( v60 != -8 )
      {
        if ( v60 == -16 && !v47 )
          v47 = v13;
        v59 = v56 & (v57 + v59);
        v13 = (__int64 *)(v58 + 72LL * v59);
        v60 = *v13;
        if ( *v70 == *v13 )
          goto LABEL_20;
        ++v57;
      }
      goto LABEL_37;
    }
    goto LABEL_85;
  }
LABEL_20:
  *(_DWORD *)(v6 + 496) = v37;
  if ( *v13 != -8 )
    --*(_DWORD *)(v6 + 500);
  *v13 = *v7;
  v13[1] = (__int64)(v13 + 3);
  v13[2] = 0x400000000LL;
  *(_DWORD *)v71 = *(_DWORD *)a1[2];
  *(_QWORD *)&v71[4] = *(_QWORD *)a3;
  v15 = 0;
LABEL_5:
  v16 = v13[1] + 12 * v15;
  *(_QWORD *)v16 = *(_QWORD *)v71;
  *(_DWORD *)(v16 + 8) = *(_DWORD *)&v71[8];
  ++*((_DWORD *)v13 + 4);
  v17 = (_DWORD *)a1[2];
  v18 = (*v17)++;
  v19 = *a1;
  v20 = *(_DWORD *)(*a1 + 72LL);
  v21 = *a1 + 48LL;
  if ( !v20 )
  {
    ++*(_QWORD *)(v19 + 48);
    goto LABEL_41;
  }
  v22 = *(_QWORD *)(v19 + 56);
  v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = (__int64 *)(v22 + 16LL * v23);
  v25 = *v24;
  if ( *v24 == a2 )
    goto LABEL_7;
  v68 = 1;
  v38 = 0;
  while ( v25 != -8 )
  {
    if ( v25 == -16 && !v38 )
      v38 = v24;
    v23 = (v20 - 1) & (v68 + v23);
    v24 = (__int64 *)(v22 + 16LL * v23);
    v25 = *v24;
    if ( *v24 == a2 )
      goto LABEL_7;
    ++v68;
  }
  v39 = *(_DWORD *)(v19 + 64);
  if ( v38 )
    v24 = v38;
  ++*(_QWORD *)(v19 + 48);
  v40 = v39 + 1;
  if ( 4 * v40 >= 3 * v20 )
  {
LABEL_41:
    sub_14672C0(v21, 2 * v20);
    v48 = *(_DWORD *)(v19 + 72);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(v19 + 56);
      v51 = (v48 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v40 = *(_DWORD *)(v19 + 64) + 1;
      v24 = (__int64 *)(v50 + 16LL * v51);
      v52 = *v24;
      if ( *v24 == a2 )
        goto LABEL_29;
      v53 = 1;
      v54 = 0;
      while ( v52 != -8 )
      {
        if ( !v54 && v52 == -16 )
          v54 = v24;
        v51 = v49 & (v53 + v51);
        v24 = (__int64 *)(v50 + 16LL * v51);
        v52 = *v24;
        if ( *v24 == a2 )
          goto LABEL_29;
        ++v53;
      }
LABEL_45:
      if ( v54 )
        v24 = v54;
      goto LABEL_29;
    }
LABEL_86:
    ++*(_DWORD *)(v19 + 64);
    BUG();
  }
  if ( v20 - *(_DWORD *)(v19 + 68) - v40 <= v20 >> 3 )
  {
    sub_14672C0(v21, v20);
    v61 = *(_DWORD *)(v19 + 72);
    if ( v61 )
    {
      v62 = v61 - 1;
      v63 = *(_QWORD *)(v19 + 56);
      v64 = 1;
      v54 = 0;
      v65 = (v61 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v40 = *(_DWORD *)(v19 + 64) + 1;
      v24 = (__int64 *)(v63 + 16LL * v65);
      v66 = *v24;
      if ( *v24 == a2 )
        goto LABEL_29;
      while ( v66 != -8 )
      {
        if ( v66 == -16 && !v54 )
          v54 = v24;
        v65 = v62 & (v64 + v65);
        v24 = (__int64 *)(v63 + 16LL * v65);
        v66 = *v24;
        if ( *v24 == a2 )
          goto LABEL_29;
        ++v64;
      }
      goto LABEL_45;
    }
    goto LABEL_86;
  }
LABEL_29:
  *(_DWORD *)(v19 + 64) = v40;
  if ( *v24 != -8 )
    --*(_DWORD *)(v19 + 68);
  *v24 = a2;
  *((_DWORD *)v24 + 2) = 0;
LABEL_7:
  *((_DWORD *)v24 + 2) = v18;
  v26 = (_QWORD *)a1[3];
  v27 = *a3;
  if ( *((_BYTE *)a3 + 4) )
  {
    v28 = 1LL << v27;
    v29 = 8LL * (v27 >> 6);
    v30 = (_QWORD *)(v29 + v26[3]);
    if ( (*v30 & v28) != 0 )
    {
      *v30 &= ~v28;
      v26 = (_QWORD *)a1[3];
      v28 = 1LL << *a3;
      v29 = 8LL * (*a3 >> 6);
    }
    result = (_QWORD *)(*v26 + v29);
    *result |= v28;
  }
  else
  {
    v32 = 1LL << v27;
    v33 = 8LL * (v27 >> 6);
    v34 = (_QWORD *)(v33 + *v26);
    if ( (*v34 & v32) != 0 )
    {
      *v34 &= ~v32;
      result = (_QWORD *)(*(_QWORD *)(a1[3] + 24LL) + 8LL * (*a3 >> 6));
      *result |= 1LL << *a3;
    }
    else
    {
      result = (_QWORD *)(v26[3] + v33);
      *result |= v32;
    }
  }
  return result;
}
