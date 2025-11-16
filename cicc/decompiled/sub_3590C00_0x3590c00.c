// Function: sub_3590C00
// Address: 0x3590c00
//
__int64 *__fastcall sub_3590C00(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rax
  unsigned __int64 *v8; // rax
  int v10; // eax
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r8
  int *v15; // r12
  int *v16; // r15
  int v17; // eax
  __int64 v18; // rdx
  unsigned __int64 v19; // r9
  __int64 v20; // rax
  int *v21; // rdx
  int v22; // eax
  __int64 *v23; // r12
  __int64 v24; // r15
  __int64 *v25; // r15
  int v26; // eax
  __int64 v27; // r8
  __int64 v28; // rdx
  unsigned __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // r9
  int v36; // eax
  __int64 v37; // r8
  __int64 v38; // rdx
  unsigned __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // r9
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // r13
  char v48; // al
  __int64 v49; // rcx
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rax
  char v52; // si
  unsigned __int64 v53; // rdx
  __int64 v54; // rdx
  _QWORD *v55; // rax
  unsigned int v56; // eax
  void *v57; // rax
  __int64 *v58; // rsi
  _QWORD *v59; // rax
  int v60; // eax
  _QWORD *v61; // rax
  unsigned __int64 *v62; // rax
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rcx
  __int64 v66; // [rsp+0h] [rbp-140h]
  int v68; // [rsp+10h] [rbp-130h]
  __int64 v69; // [rsp+10h] [rbp-130h]
  int v70; // [rsp+10h] [rbp-130h]
  int v71; // [rsp+10h] [rbp-130h]
  int v72; // [rsp+10h] [rbp-130h]
  __int64 v73; // [rsp+10h] [rbp-130h]
  int v74; // [rsp+10h] [rbp-130h]
  int v75; // [rsp+10h] [rbp-130h]
  int *v76; // [rsp+18h] [rbp-128h]
  int v77; // [rsp+18h] [rbp-128h]
  int v78; // [rsp+18h] [rbp-128h]
  char v79; // [rsp+33h] [rbp-10Dh] BYREF
  int v80; // [rsp+34h] [rbp-10Ch] BYREF
  _QWORD *v81; // [rsp+38h] [rbp-108h] BYREF
  _QWORD *v82; // [rsp+40h] [rbp-100h] BYREF
  __int64 v83; // [rsp+48h] [rbp-F8h]
  int v84; // [rsp+50h] [rbp-F0h]
  __int16 v85; // [rsp+54h] [rbp-ECh]
  char v86; // [rsp+56h] [rbp-EAh]
  unsigned __int64 v87[2]; // [rsp+60h] [rbp-E0h] BYREF
  _BYTE v88[16]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD v89[6]; // [rsp+80h] [rbp-C0h] BYREF
  unsigned __int64 *v90; // [rsp+B0h] [rbp-90h]
  _QWORD *v91; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v92; // [rsp+C8h] [rbp-78h]
  _QWORD v93[14]; // [rsp+D0h] [rbp-70h] BYREF

  v3 = a1;
  v87[0] = (unsigned __int64)v88;
  v89[5] = 0x100000000LL;
  v89[0] = &unk_49DD210;
  v87[1] = 0;
  v88[0] = 0;
  memset(&v89[1], 0, 32);
  v90 = v87;
  sub_CB5980((__int64)v89, 0, 0, 0);
  if ( (_BYTE)qword_503F748 )
  {
    v7 = (_QWORD *)sub_3572DD0(a3, 1, 1, 1, v5, v6);
    BYTE6(v93[0]) = 0;
    v91 = v7;
    WORD2(v93[0]) = 257;
    v92 = 0;
    LODWORD(v93[0]) = 16;
    sub_CB6AF0((__int64)v89, (__int64)&v91);
    v8 = v90;
    *a1 = (__int64)(a1 + 2);
    sub_35907E0(a1, (_BYTE *)*v8, *v8 + v8[1]);
    goto LABEL_3;
  }
  v10 = *(_DWORD *)(a3 + 44);
  v11 = *(_QWORD *)(a3 + 32);
  v12 = *(unsigned __int16 *)(a3 + 68);
  v91 = v93;
  v93[0] = v12 | ((unsigned __int64)(v10 & 0xFFFFFF) << 32);
  v92 = 0x1000000002LL;
  v76 = (int *)(v11 + 40LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF));
  v13 = 5LL * (unsigned int)sub_2E88FE0(a3);
  if ( v76 != (int *)(v11 + 8 * v13) )
  {
    v66 = a3;
    v15 = (int *)(v11 + 8 * v13);
    v16 = v76;
    while ( 2 )
    {
      switch ( *(_BYTE *)v15 )
      {
        case 0:
          v17 = v15[2];
          if ( v17 < 0 )
            v17 = *(unsigned __int16 *)(sub_2EBEE10(*a2, v17) + 68);
          goto LABEL_10;
        case 1:
          v17 = v15[6];
          goto LABEL_10;
        case 2:
          v54 = *((_QWORD *)v15 + 3);
          v55 = *(_QWORD **)(v54 + 24);
          if ( *(_DWORD *)(v54 + 32) > 0x40u )
            v55 = (_QWORD *)*v55;
          v82 = v55;
          v56 = *v15;
          LOBYTE(v80) = 2;
          LODWORD(v81) = (v56 >> 8) & 0xFFF;
          v17 = sub_2EADC40((char *)&v80, (int *)&v81, (__int64 *)&v82);
          goto LABEL_10;
        case 3:
          v57 = sub_C33340();
          v58 = (__int64 *)(*((_QWORD *)v15 + 3) + 24LL);
          if ( (void *)*v58 == v57 )
            sub_C3E660((__int64)&v82, (__int64)v58);
          else
            sub_C3A850((__int64)&v82, v58);
          v59 = v82;
          if ( (unsigned int)v83 > 0x40 )
            v59 = (_QWORD *)*v82;
          v81 = v59;
          v60 = 0;
          if ( *(_BYTE *)v15 )
            v60 = ((unsigned int)*v15 >> 8) & 0xFFF;
          v79 = *(_BYTE *)v15;
          v80 = v60;
          v17 = sub_2EADC40(&v79, &v80, (__int64 *)&v81);
          if ( (unsigned int)v83 > 0x40 && v82 )
          {
            v78 = v17;
            j_j___libc_free_0_0((unsigned __int64)v82);
            v17 = v78;
          }
          goto LABEL_10;
        case 4:
        case 9:
        case 0xA:
        case 0xB:
        case 0xC:
        case 0xD:
        case 0xE:
        case 0xF:
        case 0x10:
        case 0x11:
        case 0x12:
        case 0x13:
        case 0x14:
          v17 = 0;
          goto LABEL_10;
        case 5:
        case 6:
        case 8:
          v17 = sub_2EAE040(v15);
          v18 = (unsigned int)v92;
          v19 = (unsigned int)v92 + 1LL;
          if ( v19 > HIDWORD(v92) )
            goto LABEL_43;
          goto LABEL_11;
        case 7:
          v17 = v15[2] | (*v15 << 8) & 0xFFF0000;
LABEL_10:
          v18 = (unsigned int)v92;
          v19 = (unsigned int)v92 + 1LL;
          if ( v19 > HIDWORD(v92) )
          {
LABEL_43:
            v77 = v17;
            sub_C8D5F0((__int64)&v91, v93, v19, 4u, v14, v19);
            v18 = (unsigned int)v92;
            v17 = v77;
          }
LABEL_11:
          v15 += 10;
          *((_DWORD *)v91 + v18) = v17;
          LODWORD(v92) = v92 + 1;
          if ( v15 != v16 )
            continue;
          a3 = v66;
          break;
        default:
          BUG();
      }
      break;
    }
  }
  v20 = *(_QWORD *)(a3 + 48);
  v21 = (int *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_60;
  v22 = v20 & 7;
  if ( v22 )
  {
    if ( v22 != 3 )
      goto LABEL_60;
    v23 = (__int64 *)(v21 + 4);
    v24 = *v21;
  }
  else
  {
    *(_QWORD *)(a3 + 48) = v21;
    v24 = 1;
    v23 = (__int64 *)(a3 + 48);
  }
  if ( v23 != &v23[v24] )
  {
    v25 = &v23[v24];
    do
    {
      v47 = *v23;
      v48 = 1;
      v49 = 0x3FFFFFFFFFFFFFFFLL;
      v50 = *(_QWORD *)(*v23 + 24);
      if ( (v50 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
      {
        v51 = v50 >> 3;
        v52 = *(_BYTE *)(v47 + 24) & 2;
        if ( (*(_BYTE *)(v47 + 24) & 6) == 2 || (*(_BYTE *)(v47 + 24) & 1) != 0 )
        {
          v63 = HIDWORD(v50);
          v64 = HIWORD(v50);
          if ( v52 )
            v63 = v64;
          v65 = v63 + 7;
          v48 = 0;
          v49 = v65 >> 3;
        }
        else
        {
          v53 = HIDWORD(v50);
          if ( v52 )
            LODWORD(v53) = HIWORD(*(_QWORD *)(*v23 + 24));
          v48 = v51 & 1;
          v49 = ((unsigned __int64)((unsigned __int16)((unsigned int)*(_QWORD *)(*v23 + 24) >> 8) * (unsigned int)v53)
               + 7) >> 3;
        }
      }
      v82 = (_QWORD *)v49;
      LOBYTE(v83) = v48;
      v26 = sub_CA1930(&v82);
      v28 = (unsigned int)v92;
      v29 = (unsigned int)v92 + 1LL;
      if ( v29 > HIDWORD(v92) )
      {
        v75 = v26;
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v27, v29);
        v28 = (unsigned int)v92;
        v26 = v75;
      }
      *((_DWORD *)v91 + v28) = v26;
      LODWORD(v92) = v92 + 1;
      v30 = (unsigned int)v92;
      v31 = *(unsigned __int16 *)(v47 + 32);
      if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
      {
        v74 = *(unsigned __int16 *)(v47 + 32);
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v27, v31);
        v30 = (unsigned int)v92;
        LODWORD(v31) = v74;
      }
      *((_DWORD *)v91 + v30) = v31;
      LODWORD(v92) = v92 + 1;
      v32 = (unsigned int)v92;
      v33 = *(_QWORD *)(v47 + 8);
      if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
      {
        v73 = *(_QWORD *)(v47 + 8);
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v27, v33);
        v32 = (unsigned int)v92;
        LODWORD(v33) = v73;
      }
      *((_DWORD *)v91 + v32) = v33;
      LODWORD(v92) = v92 + 1;
      v34 = (unsigned int)v92;
      v35 = *(_BYTE *)(v47 + 37) & 0xF;
      if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
      {
        v72 = *(_BYTE *)(v47 + 37) & 0xF;
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v27, v35);
        v34 = (unsigned int)v92;
        LODWORD(v35) = v72;
      }
      *((_DWORD *)v91 + v34) = v35;
      LODWORD(v92) = v92 + 1;
      v36 = sub_2EAC1E0(v47);
      v38 = (unsigned int)v92;
      v39 = (unsigned int)v92 + 1LL;
      if ( v39 > HIDWORD(v92) )
      {
        v71 = v36;
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v37, v39);
        v38 = (unsigned int)v92;
        v36 = v71;
      }
      *((_DWORD *)v91 + v38) = v36;
      LODWORD(v92) = v92 + 1;
      v40 = (unsigned int)v92;
      v41 = *(unsigned __int8 *)(v47 + 36);
      if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
      {
        v70 = *(unsigned __int8 *)(v47 + 36);
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v37, v41);
        v40 = (unsigned int)v92;
        LODWORD(v41) = v70;
      }
      *((_DWORD *)v91 + v40) = v41;
      LODWORD(v92) = v92 + 1;
      v42 = (unsigned int)v92;
      v43 = 1LL << *(_BYTE *)(v47 + 34);
      if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
      {
        v69 = 1LL << *(_BYTE *)(v47 + 34);
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v37, v43);
        v42 = (unsigned int)v92;
        v43 = v69;
      }
      *((_DWORD *)v91 + v42) = v43;
      LODWORD(v92) = v92 + 1;
      v44 = (unsigned int)v92;
      v45 = *(_BYTE *)(v47 + 37) >> 4;
      if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
      {
        v68 = *(_BYTE *)(v47 + 37) >> 4;
        sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 4u, v45, v43);
        v44 = (unsigned int)v92;
        LODWORD(v45) = v68;
      }
      ++v23;
      *((_DWORD *)v91 + v44) = v45;
      v46 = (unsigned int)(v92 + 1);
      LODWORD(v92) = v92 + 1;
    }
    while ( v25 != v23 );
    v3 = a1;
    goto LABEL_61;
  }
LABEL_60:
  v46 = (unsigned int)v92;
LABEL_61:
  v61 = (_QWORD *)sub_25FD4A0(v91, (__int64)v91 + 4 * v46);
  v83 = 0;
  v82 = v61;
  v84 = 16;
  v85 = 257;
  v86 = 0;
  sub_CB6AF0((__int64)v89, (__int64)&v82);
  v62 = v90;
  *v3 = (__int64)(v3 + 2);
  sub_35907E0(v3, (_BYTE *)*v62, *v62 + v62[1]);
  if ( v91 != v93 )
    _libc_free((unsigned __int64)v91);
LABEL_3:
  v89[0] = &unk_49DD210;
  sub_CB5840((__int64)v89);
  if ( (_BYTE *)v87[0] != v88 )
    j_j___libc_free_0(v87[0]);
  return v3;
}
