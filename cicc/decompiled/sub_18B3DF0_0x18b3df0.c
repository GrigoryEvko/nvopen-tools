// Function: sub_18B3DF0
// Address: 0x18b3df0
//
__int64 __fastcall sub_18B3DF0(
        _QWORD *a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // r14
  __int64 v12; // rdi
  __int64 *v13; // r12
  __int64 *v14; // r15
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r13
  _BYTE *v17; // rbx
  __int64 v18; // rcx
  unsigned __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 *v21; // rdx
  __int64 *v22; // r14
  _BOOL4 v23; // r15d
  __int64 v24; // rcx
  __int64 v25; // r12
  __int64 v26; // rax
  unsigned __int64 *v27; // rbx
  unsigned __int64 v28; // rdx
  int v29; // r9d
  unsigned __int64 *v30; // r8
  int v31; // r11d
  unsigned int v32; // eax
  unsigned __int64 *v33; // rdi
  unsigned __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  __int64 v38; // rcx
  int v39; // eax
  _QWORD *v40; // rax
  __int64 v41; // r15
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  char v48; // bl
  unsigned __int64 *v50; // r15
  unsigned __int64 *v51; // r10
  int v52; // r11d
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // r12
  unsigned __int8 *v56; // rax
  double v57; // xmm4_8
  double v58; // xmm5_8
  _QWORD *v59; // rax
  __int64 *v60; // rdx
  __int64 *v61; // r12
  _BOOL4 v62; // r14d
  __int64 v63; // rbx
  __int64 *j; // r12
  int v65; // r11d
  __int64 *v67; // [rsp+38h] [rbp-5C8h]
  _BYTE *v68; // [rsp+48h] [rbp-5B8h]
  unsigned __int8 v69; // [rsp+56h] [rbp-5AAh]
  char v70; // [rsp+57h] [rbp-5A9h]
  _QWORD *i; // [rsp+60h] [rbp-5A0h]
  unsigned __int64 v72; // [rsp+60h] [rbp-5A0h]
  char v73; // [rsp+68h] [rbp-598h]
  unsigned __int64 v74; // [rsp+70h] [rbp-590h] BYREF
  unsigned __int64 v75; // [rsp+78h] [rbp-588h] BYREF
  __int64 v76; // [rsp+80h] [rbp-580h] BYREF
  __int64 v77; // [rsp+88h] [rbp-578h]
  __int64 v78; // [rsp+90h] [rbp-570h]
  __int64 v79; // [rsp+98h] [rbp-568h]
  __int64 v80; // [rsp+A0h] [rbp-560h] BYREF
  int v81; // [rsp+A8h] [rbp-558h] BYREF
  _QWORD *v82; // [rsp+B0h] [rbp-550h]
  int *v83; // [rsp+B8h] [rbp-548h]
  int *v84; // [rsp+C0h] [rbp-540h]
  __int64 v85; // [rsp+C8h] [rbp-538h]
  __int64 *v86; // [rsp+D0h] [rbp-530h] BYREF
  __int64 v87; // [rsp+D8h] [rbp-528h] BYREF
  __int64 *v88; // [rsp+E0h] [rbp-520h] BYREF
  __int64 *v89; // [rsp+E8h] [rbp-518h]
  __int64 *v90; // [rsp+F0h] [rbp-510h]
  __int64 v91; // [rsp+F8h] [rbp-508h]
  __int64 *v92; // [rsp+100h] [rbp-500h] BYREF
  __int64 v93; // [rsp+108h] [rbp-4F8h]
  _BYTE v94[512]; // [rsp+110h] [rbp-4F0h] BYREF
  _BYTE *v95; // [rsp+310h] [rbp-2F0h] BYREF
  __int64 v96; // [rsp+318h] [rbp-2E8h]
  _BYTE v97[64]; // [rsp+320h] [rbp-2E0h] BYREF
  _BYTE *v98; // [rsp+360h] [rbp-2A0h]
  __int64 v99; // [rsp+368h] [rbp-298h]
  _BYTE v100[64]; // [rsp+370h] [rbp-290h] BYREF
  _BYTE *v101; // [rsp+3B0h] [rbp-250h]
  __int64 v102; // [rsp+3B8h] [rbp-248h]
  _BYTE v103[64]; // [rsp+3C0h] [rbp-240h] BYREF
  _BYTE *v104; // [rsp+400h] [rbp-200h]
  __int64 v105; // [rsp+408h] [rbp-1F8h]
  _BYTE v106[64]; // [rsp+410h] [rbp-1F0h] BYREF
  _BYTE *v107; // [rsp+450h] [rbp-1B0h]
  __int64 v108; // [rsp+458h] [rbp-1A8h]
  _BYTE v109[64]; // [rsp+460h] [rbp-1A0h] BYREF
  __int64 v110; // [rsp+4A0h] [rbp-160h]
  _BYTE *v111; // [rsp+4A8h] [rbp-158h]
  _BYTE *v112; // [rsp+4B0h] [rbp-150h]
  __int64 v113; // [rsp+4B8h] [rbp-148h]
  int v114; // [rsp+4C0h] [rbp-140h]
  _BYTE v115[312]; // [rsp+4C8h] [rbp-138h] BYREF

  v9 = (__int64 *)*a1;
  v98 = v100;
  v67 = v9;
  v101 = v103;
  v95 = v97;
  v104 = v106;
  v96 = 0x800000000LL;
  v99 = 0x800000000LL;
  v102 = 0x800000000LL;
  v105 = 0x800000000LL;
  v107 = v109;
  v10 = (__int64)a1;
  v108 = 0x800000000LL;
  v111 = v115;
  v112 = v115;
  v110 = 0;
  v113 = 32;
  v114 = 0;
  sub_15ABED0((__int64)&v95, (__int64)a1);
  v11 = (_QWORD *)a1[2];
  v76 = 0;
  v92 = (__int64 *)v94;
  v93 = 0x4000000000LL;
  v83 = &v81;
  v84 = &v81;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v81 = 0;
  v82 = 0;
  v85 = 0;
  for ( i = a1 + 1; i != v11; v11 = (_QWORD *)v11[1] )
  {
    v12 = (__int64)(v11 - 7);
    v10 = (__int64)&v86;
    if ( !v11 )
      v12 = 0;
    v86 = (__int64 *)&v88;
    v87 = 0x100000000LL;
    sub_1626700(v12, (__int64)&v86);
    v13 = &v86[(unsigned int)v87];
    if ( v86 != v13 )
    {
      v14 = v86;
      do
      {
        v15 = *v14;
        v10 = (__int64)&v75;
        ++v14;
        v75 = v15;
        sub_18B3A80(&v80, &v75);
      }
      while ( v13 != v14 );
      v13 = v86;
    }
    if ( v13 != (__int64 *)&v88 )
      _libc_free((unsigned __int64)v13);
  }
  v16 = (unsigned __int64)v98;
  LODWORD(v87) = 0;
  v89 = &v87;
  v90 = &v87;
  v88 = 0;
  v91 = 0;
  v17 = &v98[8 * (unsigned int)v99];
  if ( v98 != v17 )
  {
    do
    {
      v18 = *(unsigned int *)(*(_QWORD *)v16 + 8LL);
      v19 = *(_QWORD *)(*(_QWORD *)v16 + 8 * (5 - v18));
      if ( v19 )
      {
        v10 = (__int64)&v75;
        v75 = *(_QWORD *)(*(_QWORD *)v16 + 8 * (5 - v18));
        v20 = sub_18B3BA0((__int64)&v86, &v75);
        v22 = v21;
        if ( v21 )
        {
          v23 = v20 || v21 == &v87 || v19 < v21[4];
          v10 = sub_22077B0(40);
          *(_QWORD *)(v10 + 32) = v75;
          sub_220F040(v23, v10, v22, &v87);
          ++v91;
        }
      }
      v16 += 8LL;
    }
    while ( v17 != (_BYTE *)v16 );
  }
  v68 = &v95[8 * (unsigned int)v96];
  if ( v95 == v68 )
  {
    v69 = 0;
  }
  else
  {
    v72 = (unsigned __int64)v95;
    v70 = 0;
    v69 = 0;
    do
    {
      v24 = *(unsigned int *)(*(_QWORD *)v72 + 8LL);
      v74 = *(_QWORD *)v72;
      v25 = *(_QWORD *)(v74 + 8 * (6 - v24));
      if ( v25 )
      {
        v73 = 0;
        v26 = 8LL * *(unsigned int *)(v25 + 8);
        v27 = (unsigned __int64 *)(v25 - v26);
        if ( v25 != v25 - v26 )
        {
          while ( 1 )
          {
            v75 = *v27;
            v35 = 1LL - *(unsigned int *)(v75 + 8);
            v36 = *(_QWORD *)(v75 + 8 * v35);
            if ( v36 && (unsigned __int8)sub_15B1550(v36, v10, v35) )
              sub_18B3A80(&v80, &v75);
            v10 = (unsigned int)v79;
            if ( !(_DWORD)v79 )
              break;
            v28 = v75;
            v29 = v77;
            v30 = 0;
            v31 = 1;
            v32 = (v79 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            v33 = (unsigned __int64 *)(v77 + 8LL * v32);
            v34 = *v33;
            if ( v75 == *v33 )
            {
LABEL_25:
              if ( (unsigned __int64 *)v25 == ++v27 )
                goto LABEL_43;
            }
            else
            {
              while ( v34 != -8 )
              {
                if ( v34 != -16 || v30 )
                  v33 = v30;
                v32 = (v79 - 1) & (v31 + v32);
                v50 = (unsigned __int64 *)(v77 + 8LL * v32);
                v34 = *v50;
                if ( v75 == *v50 )
                  goto LABEL_25;
                ++v31;
                v30 = v33;
                v33 = (unsigned __int64 *)(v77 + 8LL * v32);
              }
              if ( !v30 )
                v30 = v33;
              ++v76;
              v39 = v78 + 1;
              if ( 4 * ((int)v78 + 1) >= (unsigned int)(3 * v79) )
                goto LABEL_31;
              if ( (int)v79 - HIDWORD(v78) - v39 <= (unsigned int)v79 >> 3 )
              {
                sub_18B3C40((__int64)&v76, v79);
                if ( !(_DWORD)v79 )
                {
LABEL_117:
                  LODWORD(v78) = v78 + 1;
                  BUG();
                }
                v37 = v75;
                v10 = (unsigned int)(v79 - 1);
                v29 = v77;
                v51 = 0;
                v52 = 1;
                LODWORD(v53) = v10 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
                v30 = (unsigned __int64 *)(v77 + 8LL * (unsigned int)v53);
                v28 = *v30;
                v39 = v78 + 1;
                if ( v75 != *v30 )
                {
                  while ( v28 != -8 )
                  {
                    if ( v28 == -16 && !v51 )
                      v51 = v30;
                    v53 = (unsigned int)v10 & ((_DWORD)v53 + v52);
                    v30 = (unsigned __int64 *)(v77 + 8 * v53);
                    v28 = *v30;
                    if ( v75 == *v30 )
                      goto LABEL_33;
                    ++v52;
                  }
                  goto LABEL_84;
                }
              }
LABEL_33:
              LODWORD(v78) = v39;
              if ( *v30 != -8 )
                --HIDWORD(v78);
              *v30 = v28;
              v40 = v82;
              if ( v82 )
              {
                v41 = v75;
                v10 = (__int64)&v81;
                do
                {
                  while ( 1 )
                  {
                    v42 = v40[2];
                    v43 = v40[3];
                    if ( v40[4] >= v75 )
                      break;
                    v40 = (_QWORD *)v40[3];
                    if ( !v43 )
                      goto LABEL_40;
                  }
                  v10 = (__int64)v40;
                  v40 = (_QWORD *)v40[2];
                }
                while ( v42 );
LABEL_40:
                if ( (int *)v10 != &v81 && *(_QWORD *)(v10 + 32) <= v75 )
                {
                  v54 = (unsigned int)v93;
                  if ( (unsigned int)v93 >= HIDWORD(v93) )
                  {
                    v10 = (__int64)v94;
                    sub_16CD150((__int64)&v92, v94, 0, 8, (int)v30, v29);
                    v54 = (unsigned int)v93;
                  }
                  v92[v54] = v41;
                  LODWORD(v93) = v93 + 1;
                  goto LABEL_25;
                }
              }
              ++v27;
              v73 = 1;
              if ( (unsigned __int64 *)v25 == v27 )
                goto LABEL_43;
            }
          }
          ++v76;
LABEL_31:
          sub_18B3C40((__int64)&v76, 2 * v79);
          if ( !(_DWORD)v79 )
            goto LABEL_117;
          v37 = v75;
          v10 = (unsigned int)(v79 - 1);
          v29 = v77;
          LODWORD(v38) = v10 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
          v30 = (unsigned __int64 *)(v77 + 8LL * (unsigned int)v38);
          v28 = *v30;
          v39 = v78 + 1;
          if ( v75 != *v30 )
          {
            v65 = 1;
            v51 = 0;
            while ( v28 != -8 )
            {
              if ( !v51 && v28 == -16 )
                v51 = v30;
              v38 = (unsigned int)v10 & ((_DWORD)v38 + v65);
              v30 = (unsigned __int64 *)(v77 + 8 * v38);
              v28 = *v30;
              if ( v75 == *v30 )
                goto LABEL_33;
              ++v65;
            }
LABEL_84:
            v28 = v37;
            if ( v51 )
              v30 = v51;
            goto LABEL_33;
          }
          goto LABEL_33;
        }
      }
      else
      {
        v73 = 0;
      }
LABEL_43:
      if ( (_DWORD)v93 )
      {
        v10 = (__int64)&v74;
        v59 = sub_18B3BA0((__int64)&v86, &v74);
        v61 = v60;
        if ( v60 )
        {
          v62 = v59 || v60 == &v87 || v74 < v60[4];
          v10 = sub_22077B0(40);
          *(_QWORD *)(v10 + 32) = v74;
          sub_220F040(v62, v10, v61, &v87);
          ++v91;
        }
      }
      else
      {
        v44 = v88;
        if ( !v88 )
          goto LABEL_95;
        v10 = v74;
        v45 = &v87;
        do
        {
          while ( 1 )
          {
            v46 = v44[2];
            v47 = v44[3];
            if ( v44[4] >= v74 )
              break;
            v44 = (__int64 *)v44[3];
            if ( !v47 )
              goto LABEL_49;
          }
          v45 = v44;
          v44 = (__int64 *)v44[2];
        }
        while ( v46 );
LABEL_49:
        if ( v45 == &v87 )
        {
LABEL_95:
          v70 = 1;
        }
        else
        {
          v48 = v70;
          if ( v45[4] > v74 )
            v48 = 1;
          v70 = v48;
        }
      }
      if ( v73 )
      {
        v55 = v74;
        v56 = (unsigned __int8 *)sub_1627350(v67, v92, (__int64 *)(unsigned int)v93, 0, 1);
        v10 = 6;
        sub_1630830(v55, 6u, v56, a2, a3, a4, a5, v57, v58, a8, a9);
        v69 = v73;
      }
      v72 += 8LL;
      LODWORD(v93) = 0;
    }
    while ( v68 != (_BYTE *)v72 );
    if ( v70 )
    {
      v63 = sub_1632440((__int64)a1, "llvm.dbg.cu", 0xBu);
      sub_161F550(v63);
      if ( v91 )
      {
        for ( j = v89; j != &v87; j = (__int64 *)sub_220EF30(j) )
          sub_1623CA0(v63, j[4]);
      }
      v69 = v70;
    }
  }
  sub_18B21A0((__int64)v88);
  sub_18B1FD0((__int64)v82);
  j___libc_free_0(v77);
  if ( v92 != (__int64 *)v94 )
    _libc_free((unsigned __int64)v92);
  if ( v112 != v111 )
    _libc_free((unsigned __int64)v112);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  if ( v101 != v103 )
    _libc_free((unsigned __int64)v101);
  if ( v98 != v100 )
    _libc_free((unsigned __int64)v98);
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
  return v69;
}
