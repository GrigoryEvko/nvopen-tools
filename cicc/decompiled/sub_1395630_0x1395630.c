// Function: sub_1395630
// Address: 0x1395630
//
__int64 __fastcall sub_1395630(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r9d
  __int64 *v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // r13
  char *v21; // rdi
  __int64 v22; // rax
  int v24; // eax
  int v25; // edx
  __int64 v26; // rcx
  char v27; // al
  int v28; // r9d
  __int64 *v29; // r15
  int v30; // ecx
  int v31; // ecx
  int v32; // eax
  int v33; // edx
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rsi
  int v37; // r9d
  __int64 *v38; // r8
  int v39; // edx
  int v40; // edx
  __int64 v41; // rdi
  int v42; // r9d
  unsigned int v43; // eax
  __int64 v44; // rsi
  int v45; // eax
  int v46; // esi
  __int64 v47; // r9
  unsigned int v48; // eax
  int v49; // edi
  __int64 *v50; // r10
  int v51; // eax
  int v52; // r9d
  __int64 v53; // r8
  __int64 *v54; // r10
  unsigned int v55; // eax
  int v56; // edi
  unsigned int v57; // [rsp+Ch] [rbp-1E4h]
  __int64 v58; // [rsp+10h] [rbp-1E0h] BYREF
  __int64 v59; // [rsp+18h] [rbp-1D8h]
  __int64 v60; // [rsp+20h] [rbp-1D0h]
  __int64 v61; // [rsp+28h] [rbp-1C8h]
  __int64 v62; // [rsp+30h] [rbp-1C0h]
  __int64 v63; // [rsp+38h] [rbp-1B8h]
  __int64 v64; // [rsp+40h] [rbp-1B0h]
  char *v65; // [rsp+48h] [rbp-1A8h] BYREF
  char *v66; // [rsp+50h] [rbp-1A0h] BYREF
  int v67; // [rsp+58h] [rbp-198h] BYREF
  char v68; // [rsp+60h] [rbp-190h] BYREF
  char *v69; // [rsp+118h] [rbp-D8h] BYREF
  char *v70; // [rsp+120h] [rbp-D0h] BYREF
  int v71; // [rsp+128h] [rbp-C8h] BYREF
  char v72; // [rsp+130h] [rbp-C0h] BYREF
  char v73; // [rsp+1B0h] [rbp-40h]

  v2 = a1 + 16;
  v58 = a2;
  v5 = *(_DWORD *)(a1 + 40);
  v73 = 0;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_76;
  }
  v6 = *(_QWORD *)(a1 + 24);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 424LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    goto LABEL_3;
  while ( v11 != -8 )
  {
    if ( v11 != -16 || v8 )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    v11 = *(_QWORD *)(v6 + 424LL * v9);
    if ( v11 == a2 )
      goto LABEL_3;
    v8 = v10;
    ++v7;
    v10 = (__int64 *)(v6 + 424LL * v9);
  }
  if ( !v8 )
    v8 = v10;
  v24 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v5 )
  {
LABEL_76:
    sub_1390E40(v2, 2 * v5);
    v45 = *(_DWORD *)(a1 + 40);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a1 + 24);
      v48 = (v45 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v25 = *(_DWORD *)(a1 + 32) + 1;
      v8 = (__int64 *)(v47 + 424LL * v48);
      v26 = *v8;
      if ( v58 != *v8 )
      {
        v49 = 1;
        v50 = 0;
        while ( v26 != -8 )
        {
          if ( !v50 && v26 == -16 )
            v50 = v8;
          v48 = v46 & (v49 + v48);
          v8 = (__int64 *)(v47 + 424LL * v48);
          v26 = *v8;
          if ( v58 == *v8 )
            goto LABEL_30;
          ++v49;
        }
        v26 = v58;
        if ( v50 )
          v8 = v50;
      }
      goto LABEL_30;
    }
    goto LABEL_112;
  }
  v26 = a2;
  if ( v5 - *(_DWORD *)(a1 + 36) - v25 <= v5 >> 3 )
  {
    sub_1390E40(v2, v5);
    v51 = *(_DWORD *)(a1 + 40);
    if ( v51 )
    {
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 24);
      v54 = 0;
      v55 = (v51 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v25 = *(_DWORD *)(a1 + 32) + 1;
      v8 = (__int64 *)(v53 + 424LL * v55);
      v56 = 1;
      v26 = *v8;
      if ( v58 != *v8 )
      {
        while ( v26 != -8 )
        {
          if ( !v54 && v26 == -16 )
            v54 = v8;
          v55 = v52 & (v56 + v55);
          v8 = (__int64 *)(v53 + 424LL * v55);
          v26 = *v8;
          if ( v58 == *v8 )
            goto LABEL_30;
          ++v56;
        }
        v26 = v58;
        if ( v54 )
          v8 = v54;
      }
      goto LABEL_30;
    }
LABEL_112:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
LABEL_30:
  *(_DWORD *)(a1 + 32) = v25;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 36);
  *v8 = v26;
  v27 = v73;
  *((_BYTE *)v8 + 416) = v73;
  if ( v27 )
  {
    v8[3] = 0;
    v8[2] = 0;
    *((_DWORD *)v8 + 8) = 0;
    v8[1] = 1;
    ++v59;
    v8[2] = v60;
    v8[3] = v61;
    *((_DWORD *)v8 + 8) = v62;
    v60 = 0;
    v61 = 0;
    LODWORD(v62) = 0;
    v8[5] = v63;
    v8[6] = v64;
    v8[7] = (__int64)v65;
    v65 = 0;
    v64 = 0;
    v63 = 0;
    v8[8] = (__int64)(v8 + 10);
    v8[9] = 0x800000000LL;
    if ( v67 )
      sub_138EE50((__int64)(v8 + 8), &v66);
    v8[34] = (__int64)(v8 + 36);
    v8[35] = 0x800000000LL;
    if ( v71 )
      sub_138ED10((__int64)(v8 + 34), &v70);
  }
  if ( v73 )
  {
    if ( v70 != &v72 )
      _libc_free((unsigned __int64)v70);
    if ( v66 != &v68 )
      _libc_free((unsigned __int64)v66);
    if ( v63 )
      j_j___libc_free_0(v63, &v65[-v63]);
    j___libc_free_0(v60);
  }
LABEL_3:
  sub_1394520((__int64)&v58, a1, a2);
  v12 = *(_DWORD *)(a1 + 40);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_60;
  }
  v13 = *(_QWORD *)(a1 + 24);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v15 = (__int64 *)(v13 + 424LL * v14);
  v16 = *v15;
  if ( *v15 != a2 )
  {
    v28 = 1;
    v29 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v29 )
        v29 = v15;
      v14 = (v12 - 1) & (v28 + v14);
      v15 = (__int64 *)(v13 + 424LL * v14);
      v16 = *v15;
      if ( *v15 == a2 )
        goto LABEL_5;
      ++v28;
    }
    v30 = *(_DWORD *)(a1 + 32);
    if ( !v29 )
      v29 = v15;
    ++*(_QWORD *)(a1 + 16);
    v31 = v30 + 1;
    if ( 4 * v31 < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 36) - v31 > v12 >> 3 )
      {
LABEL_51:
        *(_DWORD *)(a1 + 32) = v31;
        if ( *v29 != -8 )
          --*(_DWORD *)(a1 + 36);
        *v29 = a2;
        *((_BYTE *)v29 + 416) = 0;
        goto LABEL_54;
      }
      v57 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_1390E40(v2, v12);
      v39 = *(_DWORD *)(a1 + 40);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a1 + 24);
        v38 = 0;
        v42 = 1;
        v43 = v40 & v57;
        v31 = *(_DWORD *)(a1 + 32) + 1;
        v29 = (__int64 *)(v41 + 424LL * (v40 & v57));
        v44 = *v29;
        if ( *v29 == a2 )
          goto LABEL_51;
        while ( v44 != -8 )
        {
          if ( !v38 && v44 == -16 )
            v38 = v29;
          v43 = v40 & (v42 + v43);
          v29 = (__int64 *)(v41 + 424LL * v43);
          v44 = *v29;
          if ( *v29 == a2 )
            goto LABEL_51;
          ++v42;
        }
        goto LABEL_64;
      }
      goto LABEL_111;
    }
LABEL_60:
    sub_1390E40(v2, 2 * v12);
    v32 = *(_DWORD *)(a1 + 40);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 24);
      v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = *(_DWORD *)(a1 + 32) + 1;
      v29 = (__int64 *)(v34 + 424LL * v35);
      v36 = *v29;
      if ( *v29 == a2 )
        goto LABEL_51;
      v37 = 1;
      v38 = 0;
      while ( v36 != -8 )
      {
        if ( v36 == -16 && !v38 )
          v38 = v29;
        v35 = v33 & (v37 + v35);
        v29 = (__int64 *)(v34 + 424LL * v35);
        v36 = *v29;
        if ( *v29 == a2 )
          goto LABEL_51;
        ++v37;
      }
LABEL_64:
      if ( v38 )
        v29 = v38;
      goto LABEL_51;
    }
LABEL_111:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
LABEL_5:
  if ( *((_BYTE *)v15 + 416) )
  {
    j___libc_free_0(v15[2]);
    ++v15[1];
    ++v58;
    v15[2] = v59;
    v15[3] = v60;
    *((_DWORD *)v15 + 8) = v61;
    v59 = 0;
    v60 = 0;
    LODWORD(v61) = 0;
    v17 = v15[5];
    v15[5] = v62;
    v18 = v15[7];
    v15[6] = v63;
    v15[7] = v64;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    if ( v17 )
      j_j___libc_free_0(v17, v18 - v17);
    sub_138EE50((__int64)(v15 + 8), &v65);
    sub_138ED10((__int64)(v15 + 34), &v69);
    goto LABEL_9;
  }
  v29 = v15;
LABEL_54:
  v29[1] = 1;
  ++v58;
  v29[2] = v59;
  v29[3] = v60;
  *((_DWORD *)v29 + 8) = v61;
  v59 = 0;
  v60 = 0;
  LODWORD(v61) = 0;
  v29[5] = v62;
  v29[6] = v63;
  v29[7] = v64;
  v64 = 0;
  v63 = 0;
  v62 = 0;
  v29[8] = (__int64)(v29 + 10);
  v29[9] = 0x800000000LL;
  if ( (_DWORD)v66 )
    sub_138EE50((__int64)(v29 + 8), &v65);
  v29[34] = (__int64)(v29 + 36);
  v29[35] = 0x800000000LL;
  if ( (_DWORD)v70 )
    sub_138ED10((__int64)(v29 + 34), &v69);
  *((_BYTE *)v29 + 416) = 1;
LABEL_9:
  v19 = (_QWORD *)sub_22077B0(48);
  v20 = v19;
  if ( v19 )
    *v19 = 0;
  v19[2] = 2;
  v19[3] = 0;
  v19[4] = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220(v19 + 2);
  v21 = v69;
  v20[5] = a1;
  v20[1] = &unk_49E8F78;
  v22 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v20;
  *v20 = v22;
  if ( v21 != (char *)&v71 )
    _libc_free((unsigned __int64)v21);
  if ( v65 != (char *)&v67 )
    _libc_free((unsigned __int64)v65);
  if ( v62 )
    j_j___libc_free_0(v62, v64 - v62);
  return j___libc_free_0(v59);
}
