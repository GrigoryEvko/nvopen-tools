// Function: sub_DC2D20
// Address: 0xdc2d20
//
__int64 __fastcall sub_DC2D20(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  char v4; // di
  int v5; // edi
  _QWORD *v6; // rsi
  int v7; // r9d
  unsigned int v8; // ecx
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rsi
  _QWORD *v17; // r9
  _QWORD *v18; // r15
  _QWORD **v19; // r14
  _QWORD **v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rax
  _QWORD *v28; // r9
  _QWORD *v29; // r15
  _QWORD **v30; // r14
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rax
  _QWORD *v36; // r9
  _QWORD *v37; // r15
  _QWORD **v38; // r14
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  _QWORD *v43; // r9
  _QWORD *v44; // r15
  _QWORD **v45; // r14
  __int64 v46; // rax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // rax
  _QWORD *v51; // r9
  _QWORD *v52; // r15
  _QWORD **v53; // r14
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rax
  _QWORD *v59; // r9
  _QWORD *v60; // r15
  _QWORD **v61; // r14
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // r14
  __int64 v68; // rax
  _QWORD *v69; // r9
  _QWORD *v70; // r15
  _QWORD **v71; // r14
  __int64 v72; // rax
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rsi
  __int64 v79; // rsi
  int v80; // r10d
  __int64 v81; // [rsp+8h] [rbp-98h]
  __int64 v82; // [rsp+8h] [rbp-98h]
  __int64 v83; // [rsp+8h] [rbp-98h]
  __int64 v84; // [rsp+8h] [rbp-98h]
  __int64 v85; // [rsp+8h] [rbp-98h]
  __int64 v86; // [rsp+8h] [rbp-98h]
  __int64 v87; // [rsp+8h] [rbp-98h]
  _QWORD *v88; // [rsp+10h] [rbp-90h]
  _QWORD *v89; // [rsp+10h] [rbp-90h]
  _QWORD *v90; // [rsp+10h] [rbp-90h]
  _QWORD *v91; // [rsp+10h] [rbp-90h]
  _QWORD *v92; // [rsp+10h] [rbp-90h]
  _QWORD *v93; // [rsp+10h] [rbp-90h]
  _QWORD *v94; // [rsp+10h] [rbp-90h]
  char v95; // [rsp+1Fh] [rbp-81h]
  char v96; // [rsp+1Fh] [rbp-81h]
  char v97; // [rsp+1Fh] [rbp-81h]
  char v98; // [rsp+1Fh] [rbp-81h]
  char v99; // [rsp+1Fh] [rbp-81h]
  char v100; // [rsp+1Fh] [rbp-81h]
  char v101; // [rsp+1Fh] [rbp-81h]
  __int64 v102; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v103; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v104; // [rsp+40h] [rbp-60h] BYREF
  __int64 v105; // [rsp+48h] [rbp-58h]
  _QWORD v106[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = (_QWORD *)a2;
  v4 = *((_BYTE *)a1 + 16);
  v102 = a2;
  v5 = v4 & 1;
  if ( v5 )
  {
    v6 = a1 + 3;
    v7 = 3;
  }
  else
  {
    v13 = *((unsigned int *)a1 + 8);
    v6 = (_QWORD *)a1[3];
    if ( !(_DWORD)v13 )
      goto LABEL_12;
    v7 = v13 - 1;
  }
  v8 = v7 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v9 = &v6[2 * v8];
  v10 = (_QWORD *)*v9;
  if ( v2 == (_QWORD *)*v9 )
    goto LABEL_4;
  v15 = 1;
  while ( v10 != (_QWORD *)-4096LL )
  {
    v80 = v15 + 1;
    v8 = v7 & (v15 + v8);
    v9 = &v6[2 * v8];
    v10 = (_QWORD *)*v9;
    if ( v2 == (_QWORD *)*v9 )
      goto LABEL_4;
    v15 = v80;
  }
  if ( (_BYTE)v5 )
  {
    v14 = 8;
    goto LABEL_13;
  }
  v13 = *((unsigned int *)a1 + 8);
LABEL_12:
  v14 = 2 * v13;
LABEL_13:
  v9 = &v6[v14];
LABEL_4:
  v11 = 8;
  if ( !(_BYTE)v5 )
    v11 = 2LL * *((unsigned int *)a1 + 8);
  if ( v9 == &v6[v11] )
  {
    switch ( *((_WORD *)v2 + 12) )
    {
      case 0:
      case 1:
      case 0x10:
        goto LABEL_19;
      case 2:
        v78 = sub_DC2D20(a1, v2[4]);
        if ( v78 != v2[4] )
          v2 = (_QWORD *)sub_DC5200(*a1, v78, v2[5], 0);
        goto LABEL_19;
      case 3:
        v77 = sub_DC2D20(a1, v2[4]);
        if ( v77 != v2[4] )
          v2 = sub_DC2B70(*a1, v77, v2[5], 0);
        goto LABEL_19;
      case 4:
        v79 = sub_DC2D20(a1, v2[4]);
        if ( v79 != v2[4] )
          v2 = (_QWORD *)sub_DC5000(*a1, v79, v2[5], 0);
        goto LABEL_19;
      case 5:
        v104 = v106;
        v105 = 0x200000000LL;
        v59 = (_QWORD *)v2[4];
        v93 = &v59[v2[5]];
        if ( v59 == v93 )
          goto LABEL_19;
        v100 = 0;
        v60 = (_QWORD *)v2[4];
        do
        {
          v61 = (_QWORD **)*v60;
          v20 = (_QWORD **)*v60;
          v62 = sub_DC2D20(a1, *v60);
          v65 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v86 = v62;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v63, v64);
            v65 = (unsigned int)v105;
            v62 = v86;
          }
          v104[v65] = v62;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v60;
          v100 |= v104[(unsigned int)v105 - 1] != (_QWORD)v61;
        }
        while ( v93 != v60 );
        if ( v100 )
        {
          v20 = &v104;
          v66 = sub_DC7EB0(*a1, &v104, 0, 0);
          v25 = v104;
          v2 = (_QWORD *)v66;
        }
        goto LABEL_78;
      case 6:
        v104 = v106;
        v105 = 0x200000000LL;
        v69 = (_QWORD *)v2[4];
        v94 = &v69[v2[5]];
        if ( v69 == v94 )
          goto LABEL_19;
        v101 = 0;
        v70 = (_QWORD *)v2[4];
        do
        {
          v71 = (_QWORD **)*v70;
          v20 = (_QWORD **)*v70;
          v72 = sub_DC2D20(a1, *v70);
          v75 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v81 = v72;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v73, v74);
            v75 = (unsigned int)v105;
            v72 = v81;
          }
          v104[v75] = v72;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v70;
          v101 |= v104[(unsigned int)v105 - 1] != (_QWORD)v71;
        }
        while ( v94 != v70 );
        if ( v101 )
        {
          v20 = &v104;
          v76 = sub_DC8BD0(*a1, &v104, 0, 0);
          v25 = v104;
          v2 = (_QWORD *)v76;
        }
        goto LABEL_78;
      case 7:
        v67 = sub_DC2D20(a1, v2[4]);
        v68 = sub_DC2D20(a1, v2[5]);
        if ( v67 != v2[4] || v68 != v2[5] )
          v2 = (_QWORD *)sub_DCB270(*a1, v67, v68);
        goto LABEL_19;
      case 8:
        if ( v2[6] == a1[11] )
          v2 = (_QWORD *)sub_DCC620(v2, *a1);
        else
          *((_BYTE *)a1 + 97) = 1;
        goto LABEL_19;
      case 9:
        v104 = v106;
        v105 = 0x200000000LL;
        v51 = (_QWORD *)v2[4];
        v92 = &v51[v2[5]];
        if ( v51 == v92 )
          goto LABEL_19;
        v99 = 0;
        v52 = (_QWORD *)v2[4];
        do
        {
          v53 = (_QWORD **)*v52;
          v20 = (_QWORD **)*v52;
          v54 = sub_DC2D20(a1, *v52);
          v57 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v84 = v54;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v55, v56);
            v57 = (unsigned int)v105;
            v54 = v84;
          }
          v104[v57] = v54;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v52;
          v99 |= v104[(unsigned int)v105 - 1] != (_QWORD)v53;
        }
        while ( v92 != v52 );
        if ( v99 )
        {
          v20 = &v104;
          v58 = sub_DCE040(*a1, &v104);
          v25 = v104;
          v2 = (_QWORD *)v58;
        }
        goto LABEL_78;
      case 0xA:
        v104 = v106;
        v105 = 0x200000000LL;
        v43 = (_QWORD *)v2[4];
        v91 = &v43[v2[5]];
        if ( v43 == v91 )
          goto LABEL_19;
        v98 = 0;
        v44 = (_QWORD *)v2[4];
        do
        {
          v45 = (_QWORD **)*v44;
          v20 = (_QWORD **)*v44;
          v46 = sub_DC2D20(a1, *v44);
          v49 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v85 = v46;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v47, v48);
            v49 = (unsigned int)v105;
            v46 = v85;
          }
          v104[v49] = v46;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v44;
          v98 |= v104[(unsigned int)v105 - 1] != (_QWORD)v45;
        }
        while ( v91 != v44 );
        if ( v98 )
        {
          v20 = &v104;
          v50 = sub_DCDF90(*a1, &v104);
          v25 = v104;
          v2 = (_QWORD *)v50;
        }
        goto LABEL_78;
      case 0xB:
        v104 = v106;
        v105 = 0x200000000LL;
        v36 = (_QWORD *)v2[4];
        v90 = &v36[v2[5]];
        if ( v36 == v90 )
          goto LABEL_19;
        v97 = 0;
        v37 = (_QWORD *)v2[4];
        do
        {
          v38 = (_QWORD **)*v37;
          v20 = (_QWORD **)*v37;
          v39 = sub_DC2D20(a1, *v37);
          v42 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v83 = v39;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v40, v41);
            v42 = (unsigned int)v105;
            v39 = v83;
          }
          v104[v42] = v39;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v37;
          v97 |= v104[(unsigned int)v105 - 1] != (_QWORD)v38;
        }
        while ( v90 != v37 );
        v26 = 0;
        if ( v97 )
          goto LABEL_31;
        goto LABEL_78;
      case 0xC:
        v104 = v106;
        v105 = 0x200000000LL;
        v28 = (_QWORD *)v2[4];
        v89 = &v28[v2[5]];
        if ( v28 == v89 )
          goto LABEL_19;
        v96 = 0;
        v29 = (_QWORD *)v2[4];
        do
        {
          v30 = (_QWORD **)*v29;
          v20 = (_QWORD **)*v29;
          v31 = sub_DC2D20(a1, *v29);
          v34 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v82 = v31;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v32, v33);
            v34 = (unsigned int)v105;
            v31 = v82;
          }
          v104[v34] = v31;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v29;
          v96 |= v104[(unsigned int)v105 - 1] != (_QWORD)v30;
        }
        while ( v89 != v29 );
        if ( v96 )
        {
          v20 = &v104;
          v35 = sub_DCE150(*a1, &v104);
          v25 = v104;
          v2 = (_QWORD *)v35;
        }
        goto LABEL_78;
      case 0xD:
        v104 = v106;
        v105 = 0x200000000LL;
        v17 = (_QWORD *)v2[4];
        v88 = &v17[v2[5]];
        if ( v17 == v88 )
          goto LABEL_19;
        v95 = 0;
        v18 = (_QWORD *)v2[4];
        do
        {
          v19 = (_QWORD **)*v18;
          v20 = (_QWORD **)*v18;
          v21 = sub_DC2D20(a1, *v18);
          v24 = (unsigned int)v105;
          if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
          {
            v20 = (_QWORD **)v106;
            v87 = v21;
            sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 8u, v22, v23);
            v24 = (unsigned int)v105;
            v21 = v87;
          }
          v104[v24] = v21;
          v25 = v104;
          LODWORD(v105) = v105 + 1;
          ++v18;
          v95 |= v104[(unsigned int)v105 - 1] != (_QWORD)v19;
        }
        while ( v88 != v18 );
        if ( !v95 )
          goto LABEL_78;
        v26 = 1;
LABEL_31:
        v20 = &v104;
        v27 = sub_DCEE50(*a1, &v104, v26);
        v25 = v104;
        v2 = (_QWORD *)v27;
LABEL_78:
        if ( v25 != v106 )
          _libc_free(v25, v20);
LABEL_19:
        v103 = v2;
        sub_DB11F0((__int64)&v104, (__int64)(a1 + 1), &v102, (__int64 *)&v103);
        v9 = (_QWORD *)v106[0];
        break;
      case 0xE:
        v16 = sub_DC2D20(a1, v2[4]);
        if ( v16 != v2[4] )
          v2 = (_QWORD *)sub_DD3A70(*a1, v16, v2[5]);
        goto LABEL_19;
      case 0xF:
        if ( !sub_DADE90(*a1, (__int64)v2, a1[11]) )
          *((_BYTE *)a1 + 96) = 1;
        goto LABEL_19;
      default:
        BUG();
    }
  }
  return v9[1];
}
