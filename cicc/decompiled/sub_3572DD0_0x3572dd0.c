// Function: sub_3572DD0
// Address: 0x3572dd0
//
__int64 __fastcall sub_3572DD0(__int64 a1, char a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 v8; // rax
  int v9; // ecx
  int v10; // edx
  _DWORD *v11; // rsi
  int v12; // eax
  unsigned __int64 v13; // rdx
  __int64 v14; // r13
  _QWORD *v15; // rax
  int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // r13
  int v19; // eax
  int *v20; // r8
  int *v21; // r13
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r14
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int8 v27; // al
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rax
  int *v31; // rdx
  __int64 v32; // rax
  int v34; // eax
  __int64 *v35; // rbx
  __int64 v36; // r12
  __int64 *v37; // r12
  __int64 v38; // r13
  __int64 *i; // rbx
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // r9
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r9
  char v48; // r9
  __int64 v49; // rax
  __int64 v50; // r9
  __int64 v51; // r8
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned __int64 v54; // r9
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r13
  unsigned __int64 v63; // rdx
  char v64; // al
  __int64 v65; // rcx
  unsigned __int64 v66; // rax
  char v67; // si
  unsigned __int64 v68; // rdx
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // rcx
  int *v74; // [rsp+18h] [rbp-E8h]
  __int64 v75; // [rsp+18h] [rbp-E8h]
  __int64 v76; // [rsp+18h] [rbp-E8h]
  __int64 v77; // [rsp+18h] [rbp-E8h]
  __int64 v78; // [rsp+18h] [rbp-E8h]
  __int64 v79; // [rsp+18h] [rbp-E8h]
  __int64 v80; // [rsp+18h] [rbp-E8h]
  __int64 v81; // [rsp+18h] [rbp-E8h]
  __int64 v82; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v83; // [rsp+28h] [rbp-D8h]
  __int64 v84; // [rsp+30h] [rbp-D0h]
  _QWORD *v85; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v86; // [rsp+48h] [rbp-B8h]
  _BYTE v87[176]; // [rsp+50h] [rbp-B0h] BYREF

  v7 = *(_DWORD *)(a1 + 40);
  v86 = 0x1000000000LL;
  v8 = *(_QWORD *)(a1 + 48);
  v9 = 0;
  v85 = v87;
  v10 = v7 & 0xFFFFFF;
  v11 = (_DWORD *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v12 = v8 & 7;
    if ( v12 )
    {
      if ( v12 == 3 )
      {
        v13 = (unsigned int)(v10 + *v11 + 2);
        if ( v13 <= 0x10 )
          goto LABEL_5;
        goto LABEL_59;
      }
    }
    else
    {
      *(_QWORD *)(a1 + 48) = v11;
      v9 = 1;
    }
  }
  v13 = (unsigned int)(v10 + v9 + 2);
  if ( v13 <= 0x10 )
  {
LABEL_5:
    v14 = *(unsigned __int16 *)(a1 + 68);
    v15 = v87;
    goto LABEL_6;
  }
LABEL_59:
  sub_C8D5F0((__int64)&v85, v87, v13, 8u, a5, a6);
  v14 = *(unsigned __int16 *)(a1 + 68);
  v69 = (unsigned int)v86 + 1LL;
  if ( v69 > HIDWORD(v86) )
    sub_C8D5F0((__int64)&v85, v87, v69, 8u, a5, a6);
  v15 = &v85[(unsigned int)v86];
LABEL_6:
  *v15 = v14;
  v16 = *(_DWORD *)(a1 + 44);
  LODWORD(v86) = v86 + 1;
  v17 = (unsigned int)v86;
  v18 = v16 & 0xFFFFFF;
  if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
  {
    sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, a5, a6);
    v17 = (unsigned int)v86;
  }
  v85[v17] = v18;
  v19 = *(_DWORD *)(a1 + 40);
  v20 = *(int **)(a1 + 32);
  LODWORD(v86) = v86 + 1;
  v21 = v20;
  v74 = &v20[10 * (v19 & 0xFFFFFF)];
  if ( v74 != v20 )
  {
    while ( 1 )
    {
      v27 = *(_BYTE *)v21;
      if ( a2 || v27 )
        break;
      if ( (*((_BYTE *)v21 + 3) & 0x10) == 0 || v21[2] >= 0 )
        goto LABEL_13;
LABEL_17:
      v21 += 10;
      if ( v74 == v21 )
        goto LABEL_21;
    }
    if ( v27 == 6 )
    {
      v28 = *v21;
      v29 = v21[6];
      v82 = 6;
      v84 = v29;
      v83 = (v28 >> 8) & 0xFFF;
      v24 = sub_CBF760(&v82, 0x18u);
    }
    else
    {
LABEL_13:
      v24 = sub_3572590((unsigned __int8 *)v21);
      if ( !v24 )
        goto LABEL_24;
    }
    v25 = (unsigned int)v86;
    v26 = (unsigned int)v86 + 1LL;
    if ( v26 > HIDWORD(v86) )
    {
      sub_C8D5F0((__int64)&v85, v87, v26, 8u, v22, v23);
      v25 = (unsigned int)v86;
    }
    v85[v25] = v24;
    LODWORD(v86) = v86 + 1;
    goto LABEL_17;
  }
LABEL_21:
  v30 = *(_QWORD *)(a1 + 48);
  v31 = (int *)(v30 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v30 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v34 = v30 & 7;
    if ( v34 )
    {
      if ( v34 != 3 )
        goto LABEL_22;
      v35 = (__int64 *)(v31 + 4);
      v36 = *v31;
    }
    else
    {
      *(_QWORD *)(a1 + 48) = v31;
      v36 = 1;
      v35 = (__int64 *)(a1 + 48);
    }
    v37 = &v35[v36];
    if ( v37 != v35 )
    {
      v38 = *v35;
      if ( a4 )
      {
        for ( i = v35 + 1; ; ++i )
        {
          v63 = *(_QWORD *)(v38 + 24);
          v64 = 1;
          v65 = 0x3FFFFFFFFFFFFFFFLL;
          if ( (v63 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
          {
            v66 = v63 >> 3;
            v67 = *(_BYTE *)(v38 + 24) & 2;
            if ( (*(_BYTE *)(v38 + 24) & 6) == 2 || (*(_BYTE *)(v38 + 24) & 1) != 0 )
            {
              v70 = HIDWORD(v63);
              v71 = HIWORD(v63);
              if ( v67 )
                v70 = v71;
              v72 = v70 + 7;
              v64 = 0;
              v65 = v72 >> 3;
            }
            else
            {
              v68 = HIDWORD(v63);
              if ( v67 )
                LODWORD(v68) = HIWORD(*(_QWORD *)(v38 + 24));
              v64 = v66 & 1;
              v65 = ((unsigned __int64)((unsigned __int16)((unsigned int)*(_QWORD *)(v38 + 24) >> 8) * (unsigned int)v68)
                   + 7) >> 3;
            }
          }
          v82 = v65;
          LOBYTE(v83) = v64;
          LODWORD(v42) = sub_CA1930(&v82);
          v41 = (unsigned int)v86;
          v42 = (unsigned int)v42;
          v43 = (unsigned int)v86 + 1LL;
          if ( v43 > HIDWORD(v86) )
          {
            v81 = (unsigned int)v42;
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v40, v43);
            v41 = (unsigned int)v86;
            v42 = v81;
          }
          v85[v41] = v42;
          v44 = *(unsigned __int16 *)(v38 + 32);
          LODWORD(v86) = v86 + 1;
          v45 = (unsigned int)v86;
          if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
          {
            v80 = v44;
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v40, v44);
            v45 = (unsigned int)v86;
            v44 = v80;
          }
          v85[v45] = v44;
          LODWORD(v86) = v86 + 1;
          v46 = (unsigned int)v86;
          v47 = *(unsigned int *)(v38 + 8);
          if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
          {
            v79 = *(unsigned int *)(v38 + 8);
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v40, v47);
            v46 = (unsigned int)v86;
            v47 = v79;
          }
          v85[v46] = v47;
          v48 = *(_BYTE *)(v38 + 37);
          LODWORD(v86) = v86 + 1;
          v49 = (unsigned int)v86;
          v50 = v48 & 0xF;
          if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
          {
            v78 = v50;
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v40, v50);
            v49 = (unsigned int)v86;
            v50 = v78;
          }
          v85[v49] = v50;
          LODWORD(v86) = v86 + 1;
          LODWORD(v53) = sub_2EAC1E0(v38);
          v52 = (unsigned int)v86;
          v53 = (unsigned int)v53;
          v54 = (unsigned int)v86 + 1LL;
          if ( v54 > HIDWORD(v86) )
          {
            v77 = (unsigned int)v53;
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v51, v54);
            v52 = (unsigned int)v86;
            v53 = v77;
          }
          v85[v52] = v53;
          v55 = *(unsigned __int8 *)(v38 + 36);
          LODWORD(v86) = v86 + 1;
          v56 = (unsigned int)v86;
          if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
          {
            v76 = v55;
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v51, v55);
            v56 = (unsigned int)v86;
            v55 = v76;
          }
          v85[v56] = v55;
          v58 = 1LL << *(_BYTE *)(v38 + 34);
          LODWORD(v86) = v86 + 1;
          v57 = (unsigned int)v86;
          v59 = (unsigned int)v58;
          if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
          {
            v75 = (unsigned int)v58;
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v51, (unsigned int)v58);
            v57 = (unsigned int)v86;
            v59 = v75;
          }
          v85[v57] = v59;
          v61 = *(unsigned __int8 *)(v38 + 37);
          LOBYTE(v61) = (unsigned __int8)v61 >> 4;
          LODWORD(v86) = v86 + 1;
          v60 = (unsigned int)v86;
          v62 = (unsigned __int8)v61;
          if ( (unsigned __int64)(unsigned int)v86 + 1 > HIDWORD(v86) )
          {
            sub_C8D5F0((__int64)&v85, v87, (unsigned int)v86 + 1LL, 8u, v61, v59);
            v60 = (unsigned int)v86;
          }
          v85[v60] = v62;
          v32 = (unsigned int)(v86 + 1);
          LODWORD(v86) = v86 + 1;
          if ( v37 == i )
            break;
          v38 = *i;
        }
        goto LABEL_23;
      }
    }
  }
LABEL_22:
  v32 = (unsigned int)v86;
LABEL_23:
  v24 = sub_CBF760(v85, 8 * v32);
LABEL_24:
  if ( v85 != (_QWORD *)v87 )
    _libc_free((unsigned __int64)v85);
  return v24;
}
