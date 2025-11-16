// Function: sub_264DE20
// Address: 0x264de20
//
void __fastcall sub_264DE20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  __int64 *v10; // r14
  __int64 *v11; // r15
  __int64 *v12; // r14
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r13
  unsigned int v17; // esi
  __int64 v18; // rcx
  unsigned int v19; // r8d
  int v20; // r10d
  unsigned int v21; // eax
  __int64 v22; // r14
  __int64 *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r9
  __int64 v26; // r14
  int v27; // r15d
  __int64 *v28; // r12
  __int64 *v29; // r13
  __int64 *v30; // rsi
  char v31; // al
  char v32; // dl
  __int64 v33; // r9
  char v34; // r11
  size_t v35; // rbx
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r14
  unsigned __int8 *v43; // rax
  __int64 v44; // r12
  __int64 v45; // rax
  unsigned __int64 *v46; // rbx
  unsigned __int64 *v47; // r12
  unsigned __int64 v48; // rdi
  int v49; // r10d
  __int64 *v50; // r9
  int v51; // eax
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rdx
  int *v55; // r9
  __int64 v56; // r15
  __int64 v57; // r12
  int *v58; // r13
  __int64 v59; // rdx
  int v60; // eax
  __int64 v61; // rcx
  __int64 v62; // rdi
  unsigned int v63; // esi
  int *v64; // r8
  int v65; // r10d
  __int64 v66; // rcx
  __int64 v67; // rsi
  unsigned int v68; // edi
  int *v69; // rdx
  int v70; // r10d
  __int64 v71; // rcx
  __int64 i; // rdx
  __int64 v73; // rax
  int v74; // edx
  int v75; // r8d
  __int64 v76; // [rsp+8h] [rbp-338h]
  char v77; // [rsp+8h] [rbp-338h]
  _BYTE *v78; // [rsp+10h] [rbp-330h]
  __int64 v79; // [rsp+10h] [rbp-330h]
  __int64 *v80; // [rsp+10h] [rbp-330h]
  char v81; // [rsp+10h] [rbp-330h]
  int v82; // [rsp+10h] [rbp-330h]
  int v83; // [rsp+10h] [rbp-330h]
  __int64 v84; // [rsp+18h] [rbp-328h] BYREF
  _BYTE *v85; // [rsp+20h] [rbp-320h] BYREF
  size_t v86; // [rsp+28h] [rbp-318h]
  _BYTE v87[48]; // [rsp+40h] [rbp-300h] BYREF
  unsigned __int64 v88[4]; // [rsp+70h] [rbp-2D0h] BYREF
  unsigned __int64 v89[6]; // [rsp+90h] [rbp-2B0h] BYREF
  unsigned __int64 v90[4]; // [rsp+C0h] [rbp-280h] BYREF
  unsigned __int64 v91[6]; // [rsp+E0h] [rbp-260h] BYREF
  __int64 v92; // [rsp+110h] [rbp-230h] BYREF
  __int64 v93; // [rsp+118h] [rbp-228h]
  unsigned int v94; // [rsp+128h] [rbp-218h]
  unsigned __int64 v95[6]; // [rsp+130h] [rbp-210h] BYREF
  _QWORD v96[2]; // [rsp+160h] [rbp-1E0h] BYREF
  int *v97; // [rsp+170h] [rbp-1D0h]
  unsigned __int64 *v98; // [rsp+1B0h] [rbp-190h]
  unsigned int v99; // [rsp+1B8h] [rbp-188h]
  char v100; // [rsp+1C0h] [rbp-180h] BYREF

  v84 = a2;
  v7 = a3;
  sub_26463C0((__int64)v87, a3, &v84);
  if ( !v87[32] )
    return;
  v9 = (_QWORD *)v84;
  v10 = *(__int64 **)(v84 + 104);
  if ( *(__int64 **)(v84 + 96) != v10 )
  {
    v11 = *(__int64 **)(v84 + 96);
    do
    {
      v7 = *v11++;
      sub_264DE20(a4, v7, a3, a4);
    }
    while ( v10 != v11 );
    v9 = (_QWORD *)v84;
  }
  v12 = (__int64 *)v9[10];
  if ( (__int64 *)v9[9] != v12 )
  {
    v13 = (__int64 *)v9[9];
    do
    {
      v14 = *v13;
      v13 += 2;
      v7 = *(_QWORD *)(v14 + 8);
      sub_264DE20(a4, v7, a3, a4);
    }
    while ( v12 != v13 );
    v9 = (_QWORD *)v84;
  }
  if ( !v9[1] || (unsigned __int8)sub_2647CD0((__int64)v9, v7, v8) )
    return;
  v15 = v84;
  if ( *(_BYTE *)v84 )
  {
    v31 = *(_BYTE *)(v84 + 2);
    v32 = 1;
    v33 = a1[1];
    if ( v31 != 3 )
      v32 = *(_BYTE *)(v84 + 2);
    v34 = v32;
    if ( v31 != *(_BYTE *)*a1 || LODWORD(qword_4FE8FE8[8]) > 0x63 || !*(_DWORD *)(v33 + 176) )
    {
LABEL_22:
      v76 = v33;
      sub_10391D0((__int64)&v85, v34);
      v35 = v86;
      v78 = v85;
      v36 = sub_B43CB0(*(_QWORD *)(v15 + 8));
      v37 = (_QWORD *)sub_B2BE50(v36);
      v38 = sub_A78730(v37, "memprof", 7u, v78, v35);
      v39 = *(_QWORD *)(v15 + 8);
      v79 = v38;
      v40 = (__int64 *)sub_BD5C60(v39);
      *(_QWORD *)(v39 + 72) = sub_A7B440((__int64 *)(v39 + 72), v40, -1, v79);
      v41 = sub_B43CB0(*(_QWORD *)(v15 + 8));
      v80 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(v76 + 360))(*(_QWORD *)(v76 + 368), v41);
      sub_B174A0(
        (__int64)v96,
        (__int64)"memprof-context-disambiguation",
        (__int64)"MemprofAttribute",
        16,
        *(_QWORD *)(v15 + 8));
      sub_B16080((__int64)&v92, "AllocationCall", 14, *(unsigned __int8 **)(v15 + 8));
      v42 = sub_2647050((__int64)v96, (__int64)&v92);
      sub_B18290(v42, " in clone ", 0xAu);
      v43 = (unsigned __int8 *)sub_B43CB0(*(_QWORD *)(v15 + 8));
      sub_B16080((__int64)v90, "Caller", 6, v43);
      v44 = sub_23FD640(v42, (__int64)v90);
      sub_B18290(v44, " marked with memprof allocation attribute ", 0x2Au);
      sub_B16430((__int64)v88, "Attribute", 9u, v85, v86);
      v45 = sub_23FD640(v44, (__int64)v88);
      sub_1049740(v80, v45);
      sub_2240A30(v89);
      sub_2240A30(v88);
      sub_2240A30(v91);
      sub_2240A30(v90);
      sub_2240A30(v95);
      sub_2240A30((unsigned __int64 *)&v92);
      v46 = v98;
      v96[0] = &unk_49D9D40;
      v47 = &v98[10 * v99];
      if ( v98 != v47 )
      {
        do
        {
          v47 -= 10;
          v48 = v47[4];
          if ( (unsigned __int64 *)v48 != v47 + 6 )
            j_j___libc_free_0(v48);
          if ( (unsigned __int64 *)*v47 != v47 + 2 )
            j_j___libc_free_0(*v47);
        }
        while ( v46 != v47 );
        v47 = v98;
      }
      if ( v47 != (unsigned __int64 *)&v100 )
        _libc_free((unsigned __int64)v47);
      sub_2240A30((unsigned __int64 *)&v85);
      return;
    }
    v81 = v32;
    sub_264D230((__int64)&v92, v84, *a1);
    sub_22B0690(v96, &v92);
    v55 = v97;
    if ( v97 == (int *)(v93 + 4LL * v94) )
    {
      sub_2342640((__int64)&v92);
      v33 = a1[1];
      v15 = v84;
      v34 = 2;
      goto LABEL_22;
    }
    v56 = 0;
    v57 = 0;
    v58 = (int *)(v93 + 4LL * v94);
    v77 = v81;
    while ( 1 )
    {
      v59 = a1[1];
      v60 = *v55;
      v61 = *(unsigned int *)(v59 + 152);
      v62 = *(_QWORD *)(v59 + 136);
      if ( !(_DWORD)v61 )
        goto LABEL_60;
      v63 = (v61 - 1) & (37 * v60);
      v64 = (int *)(v62 + 8LL * v63);
      v65 = *v64;
      if ( v60 != *v64 )
        break;
LABEL_50:
      v66 = *(unsigned int *)(v59 + 184);
      v67 = *(_QWORD *)(v59 + 168);
      if ( (_DWORD)v66 )
      {
        v68 = (v66 - 1) & (37 * v60);
        v69 = (int *)(v67 + 32LL * v68);
        v70 = *v69;
        if ( v60 == *v69 )
        {
LABEL_52:
          if ( v69 != (int *)(v67 + 32 * v66) )
          {
            v71 = *((_QWORD *)v69 + 1);
            for ( i = *((_QWORD *)v69 + 2); i != v71; v71 += 16 )
            {
              v56 += *(_QWORD *)(v71 + 8);
              if ( *((_BYTE *)v64 + 4) == 2 )
                v57 += *(_QWORD *)(v71 + 8);
            }
          }
        }
        else
        {
          v74 = 1;
          while ( v70 != -1 )
          {
            v68 = (v66 - 1) & (v74 + v68);
            v82 = v74 + 1;
            v69 = (int *)(v67 + 32LL * v68);
            v70 = *v69;
            if ( v60 == *v69 )
              goto LABEL_52;
            v74 = v82;
          }
        }
      }
      v97 = v55 + 1;
      sub_264DDF0((__int64)v96);
      v55 = v97;
      if ( v58 == v97 )
      {
        sub_2342640((__int64)&v92);
        v73 = 5 * v57;
        v33 = a1[1];
        v15 = v84;
        v34 = 2;
        if ( 20 * v73 < (unsigned __int64)LODWORD(qword_4FE8FE8[8]) * v56 )
          v34 = v77;
        goto LABEL_22;
      }
    }
    v75 = 1;
    while ( v65 != -1 )
    {
      v63 = (v61 - 1) & (v75 + v63);
      v83 = v75 + 1;
      v64 = (int *)(v62 + 8LL * v63);
      v65 = *v64;
      if ( v60 == *v64 )
        goto LABEL_50;
      v75 = v83;
    }
LABEL_60:
    v64 = (int *)(v62 + 8 * v61);
    goto LABEL_50;
  }
  v16 = a1[2];
  v17 = *(_DWORD *)(v16 + 24);
  v18 = *(_QWORD *)(v16 + 8);
  if ( !v17 )
    return;
  v19 = v17 - 1;
  v20 = 1;
  v21 = (v17 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
  LODWORD(v22) = v21;
  v23 = (__int64 *)(v18 + 24LL * v21);
  v24 = *v23;
  v25 = *v23;
  if ( v84 != *v23 )
  {
    while ( 1 )
    {
      if ( v25 == -4096 )
        return;
      v22 = v19 & ((_DWORD)v22 + v20);
      v25 = *(_QWORD *)(v18 + 24 * v22);
      if ( v84 == v25 )
        break;
      ++v20;
    }
    v49 = 1;
    v50 = 0;
    while ( v24 != -4096 )
    {
      if ( v24 == -8192 && !v50 )
        v50 = v23;
      v21 = v19 & (v49 + v21);
      v23 = (__int64 *)(v18 + 24LL * v21);
      v24 = *v23;
      if ( v84 == *v23 )
        goto LABEL_15;
      ++v49;
    }
    if ( !v50 )
      v50 = v23;
    v96[0] = v50;
    v51 = *(_DWORD *)(v16 + 16);
    ++*(_QWORD *)v16;
    v52 = v51 + 1;
    if ( 4 * (v51 + 1) >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(v16 + 20) - v52 > v17 >> 3 )
    {
LABEL_41:
      *(_DWORD *)(v16 + 16) = v52;
      v53 = v96[0];
      if ( *(_QWORD *)v96[0] != -4096 )
        --*(_DWORD *)(v16 + 20);
      v54 = v84;
      *(_DWORD *)(v53 + 16) = 0;
      v27 = 0;
      v26 = 0;
      *(_QWORD *)(v53 + 8) = 0;
      *(_QWORD *)v53 = v54;
      v15 = v84;
      goto LABEL_16;
    }
    sub_2644FF0(v16, v17);
    sub_263DE10(v16, &v84, v96);
    v52 = *(_DWORD *)(v16 + 16) + 1;
    goto LABEL_41;
  }
LABEL_15:
  v26 = v23[1];
  v27 = *((_DWORD *)v23 + 4);
LABEL_16:
  sub_2647100(a1[1], (__int64 *)(v15 + 8), v26, v27);
  v28 = *(__int64 **)(v84 + 24);
  v29 = &v28[2 * *(unsigned int *)(v84 + 32)];
  while ( v29 != v28 )
  {
    v30 = v28;
    v28 += 2;
    sub_2647100(a1[1], v30, v26, v27);
  }
}
