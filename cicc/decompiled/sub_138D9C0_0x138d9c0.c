// Function: sub_138D9C0
// Address: 0x138d9c0
//
__int64 __fastcall sub_138D9C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r9d
  __int64 *v8; // r12
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // r13
  _QWORD *v19; // r15
  __int64 v20; // rdi
  _QWORD *v21; // rax
  _QWORD *v22; // r12
  char *v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // rbx
  _QWORD *v26; // r12
  __int64 v27; // rdi
  int v29; // eax
  int v30; // edx
  __int64 v31; // rcx
  char v32; // al
  _QWORD *v33; // r12
  __int64 v34; // rdi
  int v35; // r9d
  __int64 *v36; // r13
  int v37; // ecx
  int v38; // ecx
  int v39; // eax
  int v40; // edx
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 v43; // rdi
  int v44; // r9d
  __int64 *v45; // r8
  int v46; // edx
  int v47; // edx
  __int64 v48; // rdi
  int v49; // r9d
  unsigned int v50; // eax
  __int64 v51; // rsi
  int v52; // eax
  __int64 v53; // rsi
  int v54; // r8d
  __int64 v55; // rdi
  unsigned int v56; // eax
  int v57; // r10d
  __int64 *v58; // r9
  int v59; // eax
  int v60; // r8d
  __int64 v61; // rdi
  int v62; // r10d
  unsigned int v63; // eax
  unsigned int v64; // [rsp+Ch] [rbp-1E4h]
  __int64 v65; // [rsp+10h] [rbp-1E0h] BYREF
  _QWORD *v66; // [rsp+18h] [rbp-1D8h]
  _QWORD *v67; // [rsp+20h] [rbp-1D0h]
  __int64 v68; // [rsp+28h] [rbp-1C8h]
  __int64 v69; // [rsp+30h] [rbp-1C0h]
  __int64 v70; // [rsp+38h] [rbp-1B8h]
  __int64 v71; // [rsp+40h] [rbp-1B0h]
  __int64 v72; // [rsp+48h] [rbp-1A8h]
  char *v73; // [rsp+50h] [rbp-1A0h] BYREF
  char *v74; // [rsp+58h] [rbp-198h] BYREF
  int v75; // [rsp+60h] [rbp-190h] BYREF
  char v76; // [rsp+68h] [rbp-188h] BYREF
  char *v77; // [rsp+120h] [rbp-D0h] BYREF
  char *v78; // [rsp+128h] [rbp-C8h] BYREF
  int v79; // [rsp+130h] [rbp-C0h] BYREF
  char v80; // [rsp+138h] [rbp-B8h] BYREF
  char v81; // [rsp+1B8h] [rbp-38h]

  v2 = a1 + 16;
  v65 = a2;
  v5 = *(_DWORD *)(a1 + 40);
  v81 = 0;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_91;
  }
  v6 = *(_QWORD *)(a1 + 24);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 432LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    goto LABEL_3;
  while ( v11 != -8 )
  {
    if ( v8 || v11 != -16 )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    v11 = *(_QWORD *)(v6 + 432LL * v9);
    if ( a2 == v11 )
      goto LABEL_3;
    v8 = v10;
    ++v7;
    v10 = (__int64 *)(v6 + 432LL * v9);
  }
  if ( !v8 )
    v8 = v10;
  v29 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  v30 = v29 + 1;
  if ( 4 * (v29 + 1) >= 3 * v5 )
  {
LABEL_91:
    sub_1388C50(v2, 2 * v5);
    v52 = *(_DWORD *)(a1 + 40);
    if ( v52 )
    {
      v53 = v65;
      v54 = v52 - 1;
      v55 = *(_QWORD *)(a1 + 24);
      v56 = (v52 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
      v8 = (__int64 *)(v55 + 432LL * v56);
      v30 = *(_DWORD *)(a1 + 32) + 1;
      v31 = *v8;
      if ( v65 == *v8 )
        goto LABEL_40;
      v57 = 1;
      v58 = 0;
      while ( v31 != -8 )
      {
        if ( v31 == -16 && !v58 )
          v58 = v8;
        v56 = v54 & (v57 + v56);
        v8 = (__int64 *)(v55 + 432LL * v56);
        v31 = *v8;
        if ( v65 == *v8 )
          goto LABEL_40;
        ++v57;
      }
LABEL_95:
      v31 = v53;
      if ( v58 )
        v8 = v58;
      goto LABEL_40;
    }
LABEL_124:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
  v31 = a2;
  if ( v5 - *(_DWORD *)(a1 + 36) - v30 <= v5 >> 3 )
  {
    sub_1388C50(v2, v5);
    v59 = *(_DWORD *)(a1 + 40);
    if ( v59 )
    {
      v53 = v65;
      v60 = v59 - 1;
      v61 = *(_QWORD *)(a1 + 24);
      v58 = 0;
      v62 = 1;
      v63 = (v59 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
      v8 = (__int64 *)(v61 + 432LL * v63);
      v30 = *(_DWORD *)(a1 + 32) + 1;
      v31 = *v8;
      if ( v65 == *v8 )
        goto LABEL_40;
      while ( v31 != -8 )
      {
        if ( v31 == -16 && !v58 )
          v58 = v8;
        v63 = v60 & (v62 + v63);
        v8 = (__int64 *)(v61 + 432LL * v63);
        v31 = *v8;
        if ( v65 == *v8 )
          goto LABEL_40;
        ++v62;
      }
      goto LABEL_95;
    }
    goto LABEL_124;
  }
LABEL_40:
  *(_DWORD *)(a1 + 32) = v30;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 36);
  *v8 = v31;
  v32 = v81;
  *((_BYTE *)v8 + 424) = v81;
  if ( v32 )
  {
    v8[3] = 0;
    v8[2] = 0;
    *((_DWORD *)v8 + 8) = 0;
    v8[1] = 1;
    v66 = (_QWORD *)((char *)v66 + 1);
    v8[2] = (__int64)v67;
    v8[3] = v68;
    *((_DWORD *)v8 + 8) = v69;
    v67 = 0;
    v68 = 0;
    LODWORD(v69) = 0;
    v8[7] = 0;
    v8[6] = 0;
    *((_DWORD *)v8 + 16) = 0;
    v8[5] = 1;
    ++v70;
    v8[6] = v71;
    v8[7] = v72;
    *((_DWORD *)v8 + 16) = (_DWORD)v73;
    v71 = 0;
    v72 = 0;
    LODWORD(v73) = 0;
    v8[9] = (__int64)(v8 + 11);
    v8[10] = 0x800000000LL;
    if ( v75 )
      sub_1381800((__int64)(v8 + 9), &v74);
    v8[35] = (__int64)(v8 + 37);
    v8[36] = 0x800000000LL;
    if ( v79 )
      sub_13816C0((__int64)(v8 + 35), &v78);
  }
  if ( v81 )
  {
    if ( v78 != &v80 )
      _libc_free((unsigned __int64)v78);
    if ( v74 != &v76 )
      _libc_free((unsigned __int64)v74);
    j___libc_free_0(v71);
    if ( (_DWORD)v69 )
    {
      v33 = v67;
      do
      {
        if ( *v33 != -8 && *v33 != -16 )
        {
          v34 = v33[1];
          if ( v34 )
            j_j___libc_free_0(v34, v33[3] - v34);
        }
        v33 += 4;
      }
      while ( &v67[4 * (unsigned int)v69] != v33 );
    }
    j___libc_free_0(v67);
  }
LABEL_3:
  sub_138AAF0((__int64)&v65, a1, a2);
  v12 = *(_DWORD *)(a1 + 40);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_75;
  }
  v13 = *(_QWORD *)(a1 + 24);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v15 = (__int64 *)(v13 + 432LL * v14);
  v16 = *v15;
  if ( a2 != *v15 )
  {
    v35 = 1;
    v36 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v36 )
        v36 = v15;
      v14 = (v12 - 1) & (v35 + v14);
      v15 = (__int64 *)(v13 + 432LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        goto LABEL_5;
      ++v35;
    }
    v37 = *(_DWORD *)(a1 + 32);
    if ( !v36 )
      v36 = v15;
    ++*(_QWORD *)(a1 + 16);
    v38 = v37 + 1;
    if ( 4 * v38 < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 36) - v38 > v12 >> 3 )
      {
LABEL_66:
        *(_DWORD *)(a1 + 32) = v38;
        if ( *v36 != -8 )
          --*(_DWORD *)(a1 + 36);
        *v36 = a2;
        *((_BYTE *)v36 + 424) = 0;
        goto LABEL_69;
      }
      v64 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_1388C50(v2, v12);
      v46 = *(_DWORD *)(a1 + 40);
      if ( v46 )
      {
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 24);
        v45 = 0;
        v49 = 1;
        v50 = v47 & v64;
        v36 = (__int64 *)(v48 + 432LL * (v47 & v64));
        v38 = *(_DWORD *)(a1 + 32) + 1;
        v51 = *v36;
        if ( a2 == *v36 )
          goto LABEL_66;
        while ( v51 != -8 )
        {
          if ( !v45 && v51 == -16 )
            v45 = v36;
          v50 = v47 & (v49 + v50);
          v36 = (__int64 *)(v48 + 432LL * v50);
          v51 = *v36;
          if ( a2 == *v36 )
            goto LABEL_66;
          ++v49;
        }
        goto LABEL_79;
      }
      goto LABEL_123;
    }
LABEL_75:
    sub_1388C50(v2, 2 * v12);
    v39 = *(_DWORD *)(a1 + 40);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 24);
      v42 = (v39 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = (__int64 *)(v41 + 432LL * v42);
      v38 = *(_DWORD *)(a1 + 32) + 1;
      v43 = *v36;
      if ( a2 == *v36 )
        goto LABEL_66;
      v44 = 1;
      v45 = 0;
      while ( v43 != -8 )
      {
        if ( !v45 && v43 == -16 )
          v45 = v36;
        v42 = v40 & (v44 + v42);
        v36 = (__int64 *)(v41 + 432LL * v42);
        v43 = *v36;
        if ( a2 == *v36 )
          goto LABEL_66;
        ++v44;
      }
LABEL_79:
      if ( v45 )
        v36 = v45;
      goto LABEL_66;
    }
LABEL_123:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
LABEL_5:
  if ( *((_BYTE *)v15 + 424) )
  {
    v17 = *((unsigned int *)v15 + 8);
    if ( (_DWORD)v17 )
    {
      v18 = (_QWORD *)v15[2];
      v19 = &v18[4 * v17];
      do
      {
        if ( *v18 != -8 && *v18 != -16 )
        {
          v20 = v18[1];
          if ( v20 )
            j_j___libc_free_0(v20, v18[3] - v20);
        }
        v18 += 4;
      }
      while ( v19 != v18 );
    }
    j___libc_free_0(v15[2]);
    ++v15[1];
    ++v65;
    v15[2] = (__int64)v66;
    v15[3] = (__int64)v67;
    *((_DWORD *)v15 + 8) = v68;
    v66 = 0;
    v67 = 0;
    LODWORD(v68) = 0;
    j___libc_free_0(v15[6]);
    ++v15[5];
    ++v69;
    v15[6] = v70;
    v15[7] = v71;
    *((_DWORD *)v15 + 16) = v72;
    v70 = 0;
    v71 = 0;
    LODWORD(v72) = 0;
    sub_1381800((__int64)(v15 + 9), &v73);
    sub_13816C0((__int64)(v15 + 35), &v77);
    goto LABEL_14;
  }
  v36 = v15;
LABEL_69:
  v36[1] = 1;
  ++v65;
  v36[2] = (__int64)v66;
  v36[3] = (__int64)v67;
  *((_DWORD *)v36 + 8) = v68;
  v66 = 0;
  v67 = 0;
  LODWORD(v68) = 0;
  v36[5] = 1;
  ++v69;
  v36[6] = v70;
  v36[7] = v71;
  *((_DWORD *)v36 + 16) = v72;
  v70 = 0;
  v71 = 0;
  LODWORD(v72) = 0;
  v36[9] = (__int64)(v36 + 11);
  v36[10] = 0x800000000LL;
  if ( (_DWORD)v74 )
    sub_1381800((__int64)(v36 + 9), &v73);
  v36[35] = (__int64)(v36 + 37);
  v36[36] = 0x800000000LL;
  if ( (_DWORD)v78 )
    sub_13816C0((__int64)(v36 + 35), &v77);
  *((_BYTE *)v36 + 424) = 1;
LABEL_14:
  v21 = (_QWORD *)sub_22077B0(48);
  v22 = v21;
  if ( v21 )
    *v21 = 0;
  v21[2] = 2;
  v21[3] = 0;
  v21[4] = a2;
  if ( a2 != -8 && a2 != -16 )
    sub_164C220(v21 + 2);
  v23 = v77;
  v22[5] = a1;
  v22[1] = &unk_49E8EA0;
  v24 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v22;
  *v22 = v24;
  if ( v23 != (char *)&v79 )
    _libc_free((unsigned __int64)v23);
  if ( v73 != (char *)&v75 )
    _libc_free((unsigned __int64)v73);
  j___libc_free_0(v70);
  if ( (_DWORD)v68 )
  {
    v25 = v66;
    v26 = &v66[4 * (unsigned int)v68];
    do
    {
      if ( *v25 != -16 && *v25 != -8 )
      {
        v27 = v25[1];
        if ( v27 )
          j_j___libc_free_0(v27, v25[3] - v27);
      }
      v25 += 4;
    }
    while ( v26 != v25 );
  }
  return j___libc_free_0(v66);
}
