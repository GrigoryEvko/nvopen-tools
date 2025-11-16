// Function: sub_12B3540
// Address: 0x12b3540
//
__int64 __fastcall sub_12B3540(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // r15
  __int64 v8; // r14
  char *v9; // rax
  char *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  char *v15; // r9
  char *v16; // rdi
  char *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  char *v20; // rsi
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // r14
  __int64 v30; // rdi
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  _QWORD *v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rbx
  int v39; // r15d
  __int64 v40; // rax
  __int64 v41; // rax
  _BYTE *v42; // rdi
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 *v48; // r14
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rsi
  int *v54; // r14
  __int64 v55; // rax
  int *v56; // rdx
  int *v57; // r13
  _BOOL4 v58; // r12d
  __int64 v59; // rax
  int *v60; // r14
  __int64 v61; // rax
  int *v62; // rdx
  int *v63; // r15
  _BOOL4 v64; // r12d
  __int64 v65; // rax
  __int64 v66; // [rsp-8h] [rbp-1B8h]
  unsigned __int64 *v67; // [rsp+10h] [rbp-1A0h]
  unsigned int v68; // [rsp+18h] [rbp-198h]
  _QWORD *v69; // [rsp+18h] [rbp-198h]
  _QWORD *v71; // [rsp+30h] [rbp-180h]
  __int64 v72; // [rsp+30h] [rbp-180h]
  __int64 v73; // [rsp+38h] [rbp-178h]
  __int64 v74; // [rsp+48h] [rbp-168h] BYREF
  _QWORD v75[2]; // [rsp+50h] [rbp-160h] BYREF
  __int16 v76; // [rsp+60h] [rbp-150h]
  char *v77; // [rsp+70h] [rbp-140h]
  char *v78; // [rsp+78h] [rbp-138h]
  char *v79; // [rsp+80h] [rbp-130h]
  char *v80; // [rsp+88h] [rbp-128h]
  _BYTE *v81; // [rsp+90h] [rbp-120h] BYREF
  __int64 v82; // [rsp+98h] [rbp-118h]
  _BYTE v83[32]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v84; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v85; // [rsp+C8h] [rbp-E8h]
  __int64 v86; // [rsp+D0h] [rbp-E0h]
  __int64 v87; // [rsp+D8h] [rbp-D8h]
  __int64 v88; // [rsp+E0h] [rbp-D0h]
  __int64 v89; // [rsp+E8h] [rbp-C8h]
  __int64 v90; // [rsp+F0h] [rbp-C0h]
  __int64 v91; // [rsp+F8h] [rbp-B8h]
  __int64 v92; // [rsp+100h] [rbp-B0h]
  __int64 v93; // [rsp+108h] [rbp-A8h]
  __int64 v94; // [rsp+110h] [rbp-A0h]
  __int64 v95; // [rsp+118h] [rbp-98h]
  _QWORD v96[12]; // [rsp+120h] [rbp-90h] BYREF
  char v97; // [rsp+180h] [rbp-30h] BYREF

  v6 = *(_QWORD *)(a4 + 16);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 16LL);
  v73 = *(_QWORD *)(v7 + 16);
  v77 = sub_128F980((__int64)a2, v6);
  if ( v8 )
  {
    v78 = sub_128F980((__int64)a2, v7);
    v9 = sub_128F980((__int64)a2, v73);
  }
  else
  {
    v8 = v73;
    v73 = 0;
    v78 = sub_128F980((__int64)a2, v7);
    v9 = 0;
  }
  v79 = v9;
  v10 = sub_128F980((__int64)a2, v8);
  v11 = a2[5];
  v80 = v10;
  v12 = sub_1643360(v11);
  v81 = v83;
  v82 = 0x300000000LL;
  if ( !byte_4F92D30 && (unsigned int)sub_2207590(&byte_4F92D30) )
  {
    dword_4F92D48 = 0;
    v60 = (int *)&v84;
    v84 = 0xF590000018BLL;
    v85 = 0x18D00000001LL;
    v86 = 0x400000F59LL;
    v87 = 0xF590000018FLL;
    v88 = 0x19100000002LL;
    v89 = 0x300000F59LL;
    v90 = 0xF5A0000018CLL;
    v91 = 0x18E00000001LL;
    v92 = 0x400000F5ALL;
    v93 = 0xF5A00000190LL;
    v94 = 0x19200000002LL;
    v95 = 0x300000F5ALL;
    qword_4F92D50 = 0;
    qword_4F92D58 = (__int64)&dword_4F92D48;
    qword_4F92D60 = (__int64)&dword_4F92D48;
    qword_4F92D68 = 0;
    v72 = v12;
    v69 = a2;
    do
    {
      v61 = sub_12B3440((_QWORD *)&dword_4F92D48 - 1, (__int64)&dword_4F92D48, v60);
      v63 = v62;
      if ( v62 )
      {
        v64 = v61 || v62 == &dword_4F92D48 || *v60 < v62[8];
        v65 = sub_22077B0(48);
        *(_QWORD *)(v65 + 32) = *(_QWORD *)v60;
        *(_DWORD *)(v65 + 40) = v60[2];
        sub_220F040(v64, v65, v63, &dword_4F92D48);
        ++qword_4F92D68;
      }
      v60 += 3;
    }
    while ( v60 != (int *)v96 );
    v12 = v72;
    a2 = v69;
    __cxa_atexit((void (*)(void *))sub_12A7740, &unk_4F92D40, &qword_4A427C0);
    sub_2207640(&byte_4F92D30);
  }
  if ( !byte_4F92CF8 && (unsigned int)sub_2207590(&byte_4F92CF8) )
  {
    dword_4F92D08 = 0;
    v54 = (int *)&v84;
    v84 = 0xE3A00000152LL;
    v85 = 0x15400000001LL;
    v86 = 0x400000E3ALL;
    v87 = 0xE3A00000156LL;
    v88 = 0x15800000002LL;
    v89 = 0x300000E3ALL;
    v90 = 0xE3B00000153LL;
    v91 = 0x15500000001LL;
    v92 = 0x400000E3BLL;
    v93 = 0xE3B00000157LL;
    v94 = 0x15900000002LL;
    v95 = 0x300000E3BLL;
    v96[0] = 0x105E0000012ELL;
    v96[1] = 0x13000000001LL;
    v96[2] = 0x40000105ELL;
    v96[3] = 0x105E00000132LL;
    v96[4] = 0x13400000002LL;
    v96[5] = 0x30000105ELL;
    v96[6] = 0x10610000012FLL;
    v96[7] = 0x13100000001LL;
    v96[8] = 0x400001061LL;
    v96[9] = 0x106100000133LL;
    v96[10] = 0x13500000002LL;
    v96[11] = 0x300001061LL;
    qword_4F92D10 = 0;
    qword_4F92D18 = (__int64)&dword_4F92D08;
    qword_4F92D20 = (__int64)&dword_4F92D08;
    qword_4F92D28 = 0;
    v71 = a2;
    do
    {
      v55 = sub_12B3440((_QWORD *)&dword_4F92D08 - 1, (__int64)&dword_4F92D08, v54);
      v57 = v56;
      if ( v56 )
      {
        v58 = v55 || v56 == &dword_4F92D08 || *v54 < v56[8];
        v59 = sub_22077B0(48);
        *(_QWORD *)(v59 + 32) = *(_QWORD *)v54;
        *(_DWORD *)(v59 + 40) = v54[2];
        sub_220F040(v58, v59, v57, &dword_4F92D08);
        ++qword_4F92D28;
      }
      v54 += 3;
    }
    while ( v54 != (int *)&v97 );
    a2 = v71;
    __cxa_atexit((void (*)(void *))sub_12A7740, &unk_4F92D00, &qword_4A427C0);
    sub_2207640(&byte_4F92CF8);
  }
  v13 = &unk_4F92D00;
  if ( v73 )
    v13 = &unk_4F92D40;
  v14 = v13[2];
  v15 = (char *)(v13 + 1);
  if ( !v14 )
    goto LABEL_14;
  v16 = (char *)(v13 + 1);
  v17 = (char *)v13[2];
  do
  {
    while ( 1 )
    {
      v18 = *((_QWORD *)v17 + 2);
      v19 = *((_QWORD *)v17 + 3);
      if ( *((_DWORD *)v17 + 8) >= a3 )
        break;
      v17 = (char *)*((_QWORD *)v17 + 3);
      if ( !v19 )
        goto LABEL_12;
    }
    v16 = v17;
    v17 = (char *)*((_QWORD *)v17 + 2);
  }
  while ( v18 );
LABEL_12:
  if ( v15 != v16 && (v20 = v15, *((_DWORD *)v16 + 8) <= a3) )
  {
    do
    {
      while ( 1 )
      {
        v44 = *(_QWORD *)(v14 + 16);
        v45 = *(_QWORD *)(v14 + 24);
        if ( *(_DWORD *)(v14 + 32) >= a3 )
          break;
        v14 = *(_QWORD *)(v14 + 24);
        if ( !v45 )
          goto LABEL_42;
      }
      v20 = (char *)v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
    while ( v44 );
LABEL_42:
    if ( v15 == v20 || *((_DWORD *)v20 + 8) > a3 )
      sub_426320((__int64)"map::at");
    v21 = *((unsigned int *)v20 + 10);
    v68 = *((_DWORD *)v20 + 9);
  }
  else
  {
LABEL_14:
    v68 = 0;
    v21 = 1;
  }
  v22 = sub_1643350(a2[5]);
  v23 = sub_159C470(v22, v21, 0);
  v24 = (unsigned int)v82;
  if ( (unsigned int)v82 >= HIDWORD(v82) )
  {
    sub_16CD150(&v81, v83, 0, 8);
    v24 = (unsigned int)v82;
  }
  *(_QWORD *)&v81[8 * v24] = v23;
  v25 = 0;
  LODWORD(v82) = v82 + 1;
  while ( 1 )
  {
    v26 = (__int64)(&v77)[v25];
    if ( v26 )
      break;
LABEL_35:
    if ( ++v25 == 4 )
      goto LABEL_36;
  }
  v27 = sub_1647190(v12, *(_DWORD *)(*(_QWORD *)v26 + 8LL) >> 8);
  v76 = 257;
  if ( v27 != *(_QWORD *)v26 )
  {
    if ( *(_BYTE *)(v26 + 16) > 0x10u )
    {
      LOWORD(v86) = 257;
      v46 = sub_15FDFF0(v26, v27, &v84, 0);
      v47 = a2[7];
      v26 = v46;
      if ( v47 )
      {
        v48 = (__int64 *)a2[8];
        sub_157E9D0(v47 + 40, v46);
        v49 = *(_QWORD *)(v26 + 24);
        v50 = *v48;
        *(_QWORD *)(v26 + 32) = v48;
        v50 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v26 + 24) = v50 | v49 & 7;
        *(_QWORD *)(v50 + 8) = v26 + 24;
        *v48 = *v48 & 7 | (v26 + 24);
      }
      sub_164B780(v26, v75);
      v51 = a2[6];
      if ( v51 )
      {
        v74 = a2[6];
        sub_1623A60(&v74, v51, 2);
        v52 = v26 + 48;
        if ( *(_QWORD *)(v26 + 48) )
        {
          sub_161E7C0(v26 + 48);
          v52 = v26 + 48;
        }
        v53 = v74;
        *(_QWORD *)(v26 + 48) = v74;
        if ( v53 )
          sub_1623210(&v74, v53, v52);
      }
    }
    else
    {
      v26 = sub_15A4A70(v26, v27);
    }
  }
  (&v77)[v25] = (char *)v26;
  if ( v25 != 3 )
  {
    LOWORD(v86) = 257;
    v28 = sub_1648A60(64, 1);
    v29 = (_QWORD *)v28;
    if ( v28 )
      sub_15F9210(v28, v12, v26, 0, 0, 0);
    v30 = a2[7];
    if ( v30 )
    {
      v67 = (unsigned __int64 *)a2[8];
      sub_157E9D0(v30 + 40, v29);
      v31 = *v67;
      v32 = v29[3] & 7LL;
      v29[4] = v67;
      v31 &= 0xFFFFFFFFFFFFFFF8LL;
      v29[3] = v31 | v32;
      *(_QWORD *)(v31 + 8) = v29 + 3;
      *v67 = *v67 & 7 | (unsigned __int64)(v29 + 3);
    }
    sub_164B780(v29, &v84);
    v33 = a2[6];
    if ( v33 )
    {
      v75[0] = a2[6];
      sub_1623A60(v75, v33, 2);
      v34 = v29 + 6;
      if ( v29[6] )
      {
        sub_161E7C0(v29 + 6);
        v34 = v29 + 6;
      }
      v35 = v75[0];
      v29[6] = v75[0];
      if ( v35 )
        sub_1623210(v75, v35, v34);
    }
    v36 = (unsigned int)v82;
    if ( (unsigned int)v82 >= HIDWORD(v82) )
    {
      sub_16CD150(&v81, v83, 0, 8);
      v36 = (unsigned int)v82;
    }
    *(_QWORD *)&v81[8 * v36] = v29;
    LODWORD(v82) = v82 + 1;
    goto LABEL_35;
  }
LABEL_36:
  v37 = (_QWORD *)a2[4];
  LOWORD(v86) = 257;
  v38 = (unsigned int)v82;
  v39 = (int)v81;
  v40 = sub_126A190(v37, v68, 0, 0);
  v41 = sub_1285290(a2 + 6, *(_QWORD *)(v40 + 24), v40, v39, v38, (__int64)&v84, 0);
  sub_12A8F50(a2 + 6, v41, (__int64)v80, 0);
  v42 = v81;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v42 != v83 )
    _libc_free(v42, v66);
  return a1;
}
