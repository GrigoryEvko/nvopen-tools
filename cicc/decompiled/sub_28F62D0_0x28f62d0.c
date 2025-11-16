// Function: sub_28F62D0
// Address: 0x28f62d0
//
void __fastcall sub_28F62D0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  int v6; // r14d
  __int64 v7; // r9
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r9
  char v11; // di
  __int64 v12; // r13
  __int64 v13; // rsi
  unsigned __int8 **v14; // rax
  unsigned int v15; // r12d
  unsigned __int8 *v16; // r15
  unsigned __int8 *v17; // r13
  unsigned __int8 *v18; // r14
  unsigned __int8 *v19; // rdi
  __int64 v20; // rax
  unsigned __int8 *v21; // r8
  unsigned __int8 *v22; // rax
  __int64 v23; // r9
  __int64 v24; // r8
  unsigned __int8 *v25; // r14
  unsigned __int8 **v26; // rax
  unsigned __int8 **v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // rdx
  unsigned __int8 **v32; // rax
  unsigned __int8 **v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r9
  char v39; // al
  int v40; // eax
  __int64 *v41; // rax
  __int64 v42; // rsi
  unsigned __int8 *v43; // r14
  unsigned __int8 *v44; // rbx
  unsigned __int8 *v45; // r8
  unsigned __int8 *v46; // rdi
  __int64 v47; // r13
  __int64 v48; // rax
  unsigned __int8 *v49; // r12
  unsigned __int8 *v50; // r9
  char v51; // r12
  unsigned __int8 v52; // al
  __int64 *v53; // r14
  __int64 *v54; // r12
  __int64 v55; // rax
  unsigned __int8 *v56; // rax
  __int64 v57; // r9
  __int64 v58; // r8
  unsigned __int8 *v59; // r13
  unsigned __int8 **v60; // rax
  unsigned __int8 **v61; // rdx
  unsigned __int8 *v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  unsigned __int8 *v65; // r12
  unsigned __int8 **v66; // rax
  unsigned __int8 **v67; // rdx
  __int64 *v68; // rax
  __int64 v69; // rax
  unsigned __int64 v70; // rdx
  int v71; // eax
  __int64 *v72; // rax
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 *v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  char v78; // [rsp+7h] [rbp-159h]
  __int64 v79; // [rsp+8h] [rbp-158h]
  unsigned int v80; // [rsp+18h] [rbp-148h]
  int v81; // [rsp+1Ch] [rbp-144h]
  __int64 v82; // [rsp+28h] [rbp-138h]
  __int64 v84; // [rsp+38h] [rbp-128h]
  __int64 v85; // [rsp+38h] [rbp-128h]
  unsigned __int8 *v86; // [rsp+38h] [rbp-128h]
  __int64 v87; // [rsp+38h] [rbp-128h]
  __int64 v88; // [rsp+38h] [rbp-128h]
  __int64 v89; // [rsp+38h] [rbp-128h]
  __int64 v90; // [rsp+40h] [rbp-120h]
  __int64 v91; // [rsp+40h] [rbp-120h]
  __int64 v92; // [rsp+40h] [rbp-120h]
  __int64 v93; // [rsp+40h] [rbp-120h]
  __int64 v94; // [rsp+40h] [rbp-120h]
  __int64 v95; // [rsp+40h] [rbp-120h]
  _QWORD v97[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v98; // [rsp+60h] [rbp-100h]
  __int16 v99; // [rsp+70h] [rbp-F0h]
  __int64 *v100; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v101; // [rsp+88h] [rbp-D8h]
  _BYTE v102[64]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v103; // [rsp+D0h] [rbp-90h] BYREF
  unsigned __int8 **v104; // [rsp+D8h] [rbp-88h]
  __int64 v105; // [rsp+E0h] [rbp-80h]
  int v106; // [rsp+E8h] [rbp-78h]
  char v107; // [rsp+ECh] [rbp-74h]
  char v108; // [rsp+F0h] [rbp-70h] BYREF

  v6 = *a2;
  v7 = *(unsigned int *)(a3 + 8);
  v100 = (__int64 *)v102;
  v8 = *(_QWORD *)a3;
  v101 = 0x800000000LL;
  v81 = v6 - 29;
  v104 = (unsigned __int8 **)&v108;
  v9 = v7;
  v10 = v8 + 16 * v7;
  v11 = 1;
  v12 = v10;
  v90 = a3;
  v80 = a4;
  v78 = a4;
  v103 = 0;
  v105 = 8;
  v106 = 0;
  v107 = 1;
  if ( v8 != v10 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v8 + 8);
        if ( v11 )
          break;
LABEL_49:
        v8 += 16;
        sub_C8CC70((__int64)&v103, v13, a3, a4, a5, v10);
        v11 = v107;
        if ( v8 == v12 )
          goto LABEL_8;
      }
      v14 = v104;
      a4 = HIDWORD(v105);
      a3 = (__int64)&v104[HIDWORD(v105)];
      if ( v104 == (unsigned __int8 **)a3 )
      {
LABEL_51:
        if ( HIDWORD(v105) >= (unsigned int)v105 )
          goto LABEL_49;
        a4 = (unsigned int)(HIDWORD(v105) + 1);
        v8 += 16;
        ++HIDWORD(v105);
        *(_QWORD *)a3 = v13;
        v11 = v107;
        ++v103;
        if ( v8 == v12 )
          goto LABEL_8;
      }
      else
      {
        while ( (unsigned __int8 *)v13 != *v14 )
        {
          if ( (unsigned __int8 **)a3 == ++v14 )
            goto LABEL_51;
        }
        v8 += 16;
        if ( v8 == v12 )
        {
LABEL_8:
          v9 = *(_DWORD *)(v90 + 8);
          v10 = *(_QWORD *)v90;
          break;
        }
      }
    }
  }
  if ( v9 == 2 )
  {
    v45 = *(unsigned __int8 **)(v10 + 8);
    v50 = *(unsigned __int8 **)(v10 + 24);
    v46 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v49 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    if ( v45 == v46 && v50 == v49 )
      goto LABEL_81;
    v43 = a2;
    v47 = 0;
    v44 = 0;
LABEL_95:
    if ( v45 == v49 && v46 == v50 )
    {
      sub_B506C0(v43);
      *(_BYTE *)(a1 + 752) = 1;
      if ( !v47 )
        goto LABEL_81;
      goto LABEL_62;
    }
    v91 = (__int64)v45;
    if ( v46 == v45 )
    {
LABEL_105:
      v93 = (__int64)v50;
      if ( v50 == v49 )
      {
LABEL_113:
        v47 = (__int64)v43;
        if ( !v44 )
          v44 = v43;
        *(_BYTE *)(a1 + 752) = 1;
LABEL_62:
        v51 = 1;
        while ( 1 )
        {
          if ( !v51 )
            goto LABEL_63;
          if ( (unsigned __int8)sub_920620((__int64)a2) )
            break;
          v52 = *(_BYTE *)v47;
          *(_BYTE *)(v47 + 1) &= 1u;
          if ( v52 != 42 && (v52 != 46 || !HIBYTE(v80)) )
            goto LABEL_63;
          if ( v78 )
          {
            sub_B447F0((unsigned __int8 *)v47, 1);
            if ( !BYTE1(v80) )
              goto LABEL_63;
          }
          else if ( !BYTE1(v80) || (v80 & 0xFF0000) == 0 )
          {
            goto LABEL_63;
          }
          sub_B44850((unsigned __int8 *)v47, 1);
          if ( (unsigned __int8 *)v47 != v44 )
          {
LABEL_64:
            if ( (unsigned __int8 *)v47 == a2 )
              goto LABEL_81;
            if ( v51 )
              sub_F50F00(v47);
            goto LABEL_67;
          }
LABEL_77:
          if ( (unsigned __int8 *)v47 == a2 )
            goto LABEL_81;
          v51 = 0;
LABEL_67:
          LOWORD(v5) = 0;
          sub_B444E0((_QWORD *)v47, (__int64)(a2 + 24), v5);
          v47 = *(_QWORD *)(*(_QWORD *)(v47 + 16) + 24LL);
        }
        v71 = sub_B45210((__int64)a2);
        *(_BYTE *)(v47 + 1) &= 1u;
        sub_B45150(v47, v71);
LABEL_63:
        if ( (unsigned __int8 *)v47 != v44 )
          goto LABEL_64;
        goto LABEL_77;
      }
      v62 = sub_28ED300(v49, v81);
      v64 = v93;
      v65 = v62;
      if ( v62 )
      {
        if ( v107 )
        {
          v66 = v104;
          v67 = &v104[HIDWORD(v105)];
          if ( v104 != v67 )
          {
            while ( v65 != *v66 )
            {
              if ( v67 == ++v66 )
                goto LABEL_126;
            }
            goto LABEL_112;
          }
          goto LABEL_126;
        }
        v72 = sub_C8CA60((__int64)&v103, (__int64)v62);
        v64 = v93;
        if ( !v72 )
        {
LABEL_126:
          v73 = (unsigned int)v101;
          v74 = (unsigned int)v101 + 1LL;
          if ( v74 > HIDWORD(v101) )
          {
            v94 = v64;
            sub_C8D5F0((__int64)&v100, v102, v74, 8u, v63, v64);
            v73 = (unsigned int)v101;
            v64 = v94;
          }
          v100[v73] = (__int64)v65;
          LODWORD(v101) = v101 + 1;
        }
      }
LABEL_112:
      sub_AC2B30((__int64)(v43 - 32), v64);
      goto LABEL_113;
    }
    v87 = (__int64)v50;
    v56 = sub_28ED300(v46, v81);
    v57 = v87;
    v58 = v91;
    v59 = v56;
    if ( v56 )
    {
      if ( v107 )
      {
        v60 = v104;
        v61 = &v104[HIDWORD(v105)];
        if ( v104 != v61 )
        {
          while ( v59 != *v60 )
          {
            if ( v61 == ++v60 )
              goto LABEL_130;
          }
          goto LABEL_104;
        }
        goto LABEL_130;
      }
      v75 = sub_C8CA60((__int64)&v103, (__int64)v56);
      v58 = v91;
      v57 = v87;
      if ( !v75 )
      {
LABEL_130:
        v76 = (unsigned int)v101;
        v77 = (unsigned int)v101 + 1LL;
        if ( v77 > HIDWORD(v101) )
        {
          v89 = v57;
          v95 = v58;
          sub_C8D5F0((__int64)&v100, v102, v77, 8u, v58, v57);
          v76 = (unsigned int)v101;
          v57 = v89;
          v58 = v95;
        }
        v100[v76] = (__int64)v59;
        LODWORD(v101) = v101 + 1;
      }
    }
LABEL_104:
    v92 = v57;
    sub_AC2B30((__int64)(v43 - 64), v58);
    v50 = (unsigned __int8 *)v92;
    goto LABEL_105;
  }
  v15 = 0;
  v79 = v5;
  v16 = 0;
  v17 = a2;
  v18 = 0;
  while ( 1 )
  {
    v19 = (unsigned __int8 *)*((_QWORD *)v17 - 4);
    v20 = 16LL * v15;
    v21 = *(unsigned __int8 **)(v10 + v20 + 8);
    if ( v21 == v19 )
      goto LABEL_29;
    if ( v21 != *((unsigned __int8 **)v17 - 8) )
    {
      v84 = *(_QWORD *)(v10 + v20 + 8);
      v22 = sub_28ED300(v19, v81);
      v24 = v84;
      v25 = v22;
      if ( v22 )
      {
        if ( v107 )
        {
          v26 = v104;
          v27 = &v104[HIDWORD(v105)];
          if ( v104 != v27 )
          {
            while ( v25 != *v26 )
            {
              if ( v27 == ++v26 )
                goto LABEL_117;
            }
            goto LABEL_19;
          }
          goto LABEL_117;
        }
        v68 = sub_C8CA60((__int64)&v103, (__int64)v22);
        v24 = v84;
        if ( !v68 )
        {
LABEL_117:
          v69 = (unsigned int)v101;
          v70 = (unsigned int)v101 + 1LL;
          if ( v70 > HIDWORD(v101) )
          {
            v88 = v24;
            sub_C8D5F0((__int64)&v100, v102, v70, 8u, v24, v23);
            v69 = (unsigned int)v101;
            v24 = v88;
          }
          v100[v69] = (__int64)v25;
          LODWORD(v101) = v101 + 1;
        }
      }
LABEL_19:
      if ( *((_QWORD *)v17 - 4) )
      {
        v28 = *((_QWORD *)v17 - 3);
        **((_QWORD **)v17 - 2) = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = *((_QWORD *)v17 - 2);
      }
      *((_QWORD *)v17 - 4) = v24;
      if ( v24 )
      {
        v29 = *(_QWORD *)(v24 + 16);
        *((_QWORD *)v17 - 3) = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 16) = v17 - 24;
        *((_QWORD *)v17 - 2) = v24 + 16;
        *(_QWORD *)(v24 + 16) = v17 - 32;
      }
      v18 = v17;
      if ( !v16 )
        v16 = v17;
      goto LABEL_28;
    }
    sub_B506C0(v17);
LABEL_28:
    *(_BYTE *)(a1 + 752) = 1;
LABEL_29:
    v30 = sub_28ED300(*((unsigned __int8 **)v17 - 8), v81);
    v31 = v30;
    if ( v30 )
    {
      if ( v107 )
      {
        v32 = v104;
        v33 = &v104[HIDWORD(v105)];
        if ( v104 == v33 )
          goto LABEL_58;
        while ( v31 != *v32 )
        {
          if ( v33 == ++v32 )
            goto LABEL_58;
        }
      }
      else
      {
        v86 = v30;
        v41 = sub_C8CA60((__int64)&v103, (__int64)v30);
        v31 = v86;
        if ( !v41 )
        {
LABEL_58:
          v17 = v31;
          goto LABEL_47;
        }
      }
    }
    if ( (_DWORD)v101 )
    {
      v34 = v100[(unsigned int)v101 - 1];
      LODWORD(v101) = v101 - 1;
    }
    else
    {
      v37 = sub_ACADE0(*((__int64 ***)a2 + 1));
      v38 = v82;
      LOWORD(v38) = 0;
      v99 = 257;
      v85 = sub_B504D0(v81, v37, v37, (__int64)v97, (__int64)(a2 + 24), v38);
      v39 = sub_920620(v85);
      v34 = v85;
      if ( v39 )
      {
        v40 = sub_B45210((__int64)a2);
        sub_B45150(v85, v40);
        v34 = v85;
      }
    }
    if ( *((_QWORD *)v17 - 8) )
    {
      v35 = *((_QWORD *)v17 - 7);
      **((_QWORD **)v17 - 6) = v35;
      if ( v35 )
        *(_QWORD *)(v35 + 16) = *((_QWORD *)v17 - 6);
    }
    *((_QWORD *)v17 - 8) = v34;
    if ( v34 )
    {
      v36 = *(_QWORD *)(v34 + 16);
      *((_QWORD *)v17 - 7) = v36;
      if ( v36 )
        *(_QWORD *)(v36 + 16) = v17 - 56;
      *((_QWORD *)v17 - 6) = v34 + 16;
      *(_QWORD *)(v34 + 16) = v17 - 64;
    }
    v18 = v17;
    if ( !v16 )
      v16 = v17;
    v17 = (unsigned __int8 *)v34;
    *(_BYTE *)(a1 + 752) = 1;
LABEL_47:
    v10 = *(_QWORD *)v90;
    if ( *(_DWORD *)(v90 + 8) == v15 + 3 )
      break;
    ++v15;
  }
  v42 = (__int64)v18;
  v43 = v17;
  v44 = v16;
  v45 = *(unsigned __int8 **)(v10 + 16LL * (v15 + 1) + 8);
  v46 = (unsigned __int8 *)*((_QWORD *)v17 - 8);
  v47 = v42;
  v48 = 16LL * (v15 + 2);
  v5 = v79;
  v49 = (unsigned __int8 *)*((_QWORD *)v43 - 4);
  v50 = *(unsigned __int8 **)(v10 + v48 + 8);
  if ( v46 != v45 || v50 != v49 )
    goto LABEL_95;
  if ( v42 )
    goto LABEL_62;
LABEL_81:
  v53 = v100;
  v54 = &v100[(unsigned int)v101];
  if ( v54 != v100 )
  {
    do
    {
      v55 = *v53;
      v97[0] = 0;
      v97[1] = 0;
      v98 = v55;
      if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
        sub_BD73F0((__int64)v97);
      sub_28F19A0(a1 + 64, v97);
      if ( v98 != 0 && v98 != -4096 && v98 != -8192 )
        sub_BD60C0(v97);
      ++v53;
    }
    while ( v54 != v53 );
  }
  if ( !v107 )
    _libc_free((unsigned __int64)v104);
  if ( v100 != (__int64 *)v102 )
    _libc_free((unsigned __int64)v100);
}
