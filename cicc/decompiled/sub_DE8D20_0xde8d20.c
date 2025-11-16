// Function: sub_DE8D20
// Address: 0xde8d20
//
__int64 __fastcall sub_DE8D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // edi
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r15
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // r15
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 *v25; // r10
  __int64 v26; // rax
  unsigned int v27; // r14d
  __int64 v28; // r15
  __int64 *v29; // r13
  __int64 v30; // rbx
  __int16 v31; // ax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 v35; // r8
  _QWORD *v36; // r11
  __int64 v37; // r12
  __int64 *v38; // r15
  __int64 v39; // rbx
  __int64 v40; // r13
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  _BYTE *v43; // r12
  _BYTE *v44; // rdi
  __int64 *v45; // rax
  int v47; // eax
  __int64 *v48; // r14
  _QWORD *v49; // r13
  _QWORD *v50; // rax
  _QWORD *v51; // rsi
  __int64 v52; // r8
  __int64 v53; // r9
  _QWORD *v54; // r13
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  _QWORD *v57; // r13
  __int64 v58; // rax
  __int64 *v59; // r15
  _QWORD *v60; // r13
  __int64 v61; // rax
  _BYTE *v62; // rax
  _BYTE *v63; // r13
  _QWORD *v64; // rax
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // r14
  bool v69; // zf
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 *v73; // rax
  int v74; // ecx
  unsigned int v75; // esi
  int v76; // edx
  __int64 v77; // rdx
  __int64 v78; // rdx
  _QWORD *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  char *v84; // rdi
  _QWORD *v85; // r13
  __int64 v86; // rax
  _BYTE *v87; // rax
  int v88; // r8d
  __int64 v89; // r8
  __int64 v90; // r9
  _QWORD *v91; // r15
  __int64 v92; // rax
  unsigned __int64 v93; // rdx
  __int64 v94; // r8
  __int64 v95; // r9
  _QWORD *v96; // r13
  __int64 v97; // rax
  unsigned __int64 v98; // rdx
  __int64 v99; // [rsp+0h] [rbp-150h]
  _QWORD *v100; // [rsp+0h] [rbp-150h]
  __int64 v101; // [rsp+8h] [rbp-148h]
  __int64 v102; // [rsp+8h] [rbp-148h]
  bool v103; // [rsp+17h] [rbp-139h]
  __int64 v106; // [rsp+30h] [rbp-120h]
  __int64 v107; // [rsp+38h] [rbp-118h]
  __int64 v108; // [rsp+38h] [rbp-118h]
  __int64 v109; // [rsp+40h] [rbp-110h]
  __int64 v110; // [rsp+40h] [rbp-110h]
  int v111; // [rsp+48h] [rbp-108h]
  __int64 v112; // [rsp+48h] [rbp-108h]
  __int64 v113; // [rsp+48h] [rbp-108h]
  __int64 *v114; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v115; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v116; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+68h] [rbp-E8h]
  _BYTE *v118; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v119; // [rsp+78h] [rbp-D8h]
  _BYTE v120[32]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD *v121; // [rsp+A0h] [rbp-B0h]
  char *v122; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 v123; // [rsp+B0h] [rbp-A0h]
  _BYTE v124[24]; // [rsp+B8h] [rbp-98h] BYREF
  _BYTE *v125; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v126; // [rsp+D8h] [rbp-78h]
  _BYTE v127[112]; // [rsp+E0h] [rbp-70h] BYREF

  v3 = 0;
  v4 = *(_QWORD *)(a3 + 24);
  v118 = v120;
  v119 = 0x300000000LL;
  v106 = a2;
  if ( *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) == 12 )
  {
    v5 = *(_QWORD *)(a2 + 48);
    v6 = *(_QWORD *)(v4 + 40);
    v7 = *(_DWORD *)(v5 + 24);
    a2 = *(_QWORD *)(v5 + 8);
    if ( v7 )
    {
      v8 = v7 - 1;
      v9 = v8 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v10 = (__int64 *)(a2 + 16LL * v9);
      v11 = *v10;
      if ( v6 == *v10 )
      {
LABEL_4:
        v3 = v10[1];
        if ( v3 && v6 != **(_QWORD **)(v3 + 32) )
          v3 = 0;
      }
      else
      {
        v47 = 1;
        while ( v11 != -4096 )
        {
          v88 = v47 + 1;
          v9 = v8 & (v47 + v9);
          v10 = (__int64 *)(a2 + 16LL * v9);
          v11 = *v10;
          if ( v6 == *v10 )
            goto LABEL_4;
          v47 = v88;
        }
        v3 = 0;
      }
    }
  }
  if ( (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) == 0 )
  {
    v44 = v120;
    goto LABEL_47;
  }
  v12 = 8LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  v13 = v4;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = v13;
  do
  {
    v18 = *(_QWORD *)(v17 - 8);
    v19 = *(_QWORD *)(v18 + 4 * v14);
    a2 = *(_QWORD *)(32LL * *(unsigned int *)(v17 + 72) + v18 + v14);
    if ( *(_BYTE *)(v3 + 84) )
    {
      v20 = *(_QWORD **)(v3 + 64);
      v21 = &v20[*(unsigned int *)(v3 + 76)];
      if ( v20 != v21 )
      {
        while ( a2 != *v20 )
        {
          if ( v21 == ++v20 )
            goto LABEL_44;
        }
LABEL_14:
        if ( v15 )
        {
          if ( v19 != v15 )
            goto LABEL_46;
        }
        else
        {
          v15 = v19;
        }
        goto LABEL_16;
      }
    }
    else
    {
      v110 = v12;
      v113 = v16;
      v45 = sub_C8CA60(v3 + 56, a2);
      v16 = v113;
      v12 = v110;
      if ( v45 )
        goto LABEL_14;
    }
LABEL_44:
    if ( v16 )
    {
      if ( v19 != v16 )
        goto LABEL_46;
    }
    else
    {
      v16 = v19;
    }
LABEL_16:
    v14 += 8;
  }
  while ( v12 != v14 );
  v22 = v16;
  if ( !v15 || !v16 )
  {
LABEL_46:
    v44 = v118;
LABEL_47:
    *(_BYTE *)(a1 + 48) = 0;
    goto LABEL_48;
  }
  a2 = v15;
  v23 = sub_DD8400(v106, v15);
  v25 = v23;
  if ( *((_WORD *)v23 + 12) != 5 )
    goto LABEL_53;
  v26 = v23[5];
  v27 = v26;
  if ( (_DWORD)v26 )
  {
    v99 = v3;
    v107 = (unsigned int)v26;
    v101 = v22;
    v28 = 0;
    v29 = v25;
    v109 = a3 + 32;
    while ( 1 )
    {
      v27 = v28;
      v30 = *(_QWORD *)(v29[4] + 8 * v28);
      if ( v30 != v109 )
      {
        v111 = sub_D97050(v106, *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8LL));
        a2 = sub_D95540(v30);
        if ( v111 == (unsigned int)sub_D97050(v106, a2) )
        {
          v31 = *(_WORD *)(v30 + 24);
          if ( v31 == 4 )
          {
            v32 = *(_QWORD *)(v30 + 32);
            if ( *(_WORD *)(v32 + 24) == 2 )
              goto LABEL_26;
          }
          else if ( v31 == 3 )
          {
            v32 = *(_QWORD *)(v30 + 32);
            v30 = 0;
            if ( *(_WORD *)(v32 + 24) == 2 )
            {
LABEL_26:
              if ( v109 == *(_QWORD *)(v32 + 32) )
              {
                v33 = *(_QWORD *)(v32 + 40);
                v103 = v30 != 0;
                if ( v33 )
                {
                  v25 = v29;
                  v112 = v33;
                  v34 = v28;
                  v3 = v99;
                  v22 = v101;
                  v26 = v25[5];
                  goto LABEL_29;
                }
              }
            }
          }
        }
      }
      v27 = ++v28;
      if ( v107 == v28 )
      {
        v25 = v29;
        v22 = v101;
        v34 = v27;
        v112 = 0;
        v3 = v99;
        v26 = v25[5];
        goto LABEL_29;
      }
    }
  }
  v112 = 0;
  v34 = 0;
LABEL_29:
  if ( v26 == v34 )
  {
LABEL_53:
    *(_BYTE *)(a1 + 48) = 0;
    v44 = v118;
  }
  else
  {
    v35 = (unsigned int)v26;
    v36 = &v125;
    v37 = 0;
    v125 = v127;
    v126 = 0x800000000LL;
    if ( (_DWORD)v26 )
    {
      v108 = v22;
      v38 = v25;
      v102 = v3;
      v39 = (unsigned int)v26;
      do
      {
        if ( v27 != (_DWORD)v37 )
        {
          v40 = *(_QWORD *)(v38[4] + 8 * v37);
          v41 = (unsigned int)v126;
          v42 = (unsigned int)v126 + 1LL;
          if ( v42 > HIDWORD(v126) )
          {
            v100 = v36;
            sub_C8D5F0((__int64)v36, v127, v42, 8u, v35, v24);
            v41 = (unsigned int)v126;
            v36 = v100;
          }
          *(_QWORD *)&v125[8 * v41] = v40;
          LODWORD(v126) = v126 + 1;
        }
        ++v37;
      }
      while ( v39 != v37 );
      v22 = v108;
      v3 = v102;
    }
    a2 = (__int64)sub_DC7EB0((__int64 *)v106, (__int64)v36, 0, 0);
    v43 = (_BYTE *)a2;
    if ( !sub_DADE90(v106, a2, v3) )
    {
LABEL_39:
      *(_BYTE *)(a1 + 48) = 0;
      goto LABEL_40;
    }
    v48 = sub_DD8400(v106, v22);
    v49 = sub_DC5200(v106, a2, v112, 0);
    v50 = sub_DC5200(v106, (__int64)v48, v112, 0);
    v51 = sub_DC1960(v106, (__int64)v50, (__int64)v49, v3, 0);
    if ( *((_WORD *)v51 + 12) == 8 )
    {
      v54 = sub_DA4270(v106, (__int64)v51, 1 - ((unsigned int)!v103 - 1));
      v55 = (unsigned int)v119;
      v56 = (unsigned int)v119 + 1LL;
      if ( v56 > HIDWORD(v119) )
      {
        sub_C8D5F0((__int64)&v118, v120, v56, 8u, v52, v53);
        v55 = (unsigned int)v119;
      }
      *(_QWORD *)&v118[8 * v55] = v54;
      LODWORD(v119) = v119 + 1;
    }
    v57 = sub_DC5200(v106, (__int64)v48, v112, 0);
    v58 = sub_D95540((__int64)v48);
    if ( v103 )
      v59 = sub_DC5000(v106, (__int64)v57, v58, 0);
    else
      v59 = sub_DC2B70(v106, (__int64)v57, v58, 0);
    if ( v48 == v59 )
    {
      v85 = sub_DC5200(v106, (__int64)v43, v112, 0);
      v86 = sub_D95540((__int64)v43);
      v87 = sub_DC5000(v106, (__int64)v85, v86, 0);
      v63 = v87;
      if ( v43 != v87 )
      {
        a2 = 33;
        if ( (unsigned __int8)sub_DC3A60(v106, 33, v43, v87) )
          goto LABEL_39;
        goto LABEL_72;
      }
    }
    else
    {
      a2 = 33;
      if ( (unsigned __int8)sub_DC3A60(v106, 33, v48, v59) )
        goto LABEL_39;
      v60 = sub_DC5200(v106, (__int64)v43, v112, 0);
      v61 = sub_D95540((__int64)v43);
      v62 = sub_DC5000(v106, (__int64)v60, v61, 0);
      v63 = v62;
      if ( v62 != v43 )
      {
        a2 = 33;
        if ( (unsigned __int8)sub_DC3A60(v106, 33, v43, v62) )
          goto LABEL_39;
      }
      if ( !(unsigned __int8)sub_DC3A60(v106, 32, v48, v59) )
      {
        v91 = sub_DA4260(v106, (__int64)v48, (__int64)v59);
        v92 = (unsigned int)v119;
        v93 = (unsigned int)v119 + 1LL;
        if ( v93 > HIDWORD(v119) )
        {
          sub_C8D5F0((__int64)&v118, v120, v93, 8u, v89, v90);
          v92 = (unsigned int)v119;
        }
        *(_QWORD *)&v118[8 * v92] = v91;
        LODWORD(v119) = v119 + 1;
      }
LABEL_72:
      if ( v43 != v63 && !(unsigned __int8)sub_DC3A60(v106, 32, v43, v63) )
      {
        v96 = sub_DA4260(v106, (__int64)v43, (__int64)v63);
        v97 = (unsigned int)v119;
        v98 = (unsigned int)v119 + 1LL;
        if ( v98 > HIDWORD(v119) )
        {
          sub_C8D5F0((__int64)&v118, v120, v98, 8u, v94, v95);
          v97 = (unsigned int)v119;
        }
        *(_QWORD *)&v118[8 * v97] = v96;
        LODWORD(v119) = v119 + 1;
      }
    }
    v64 = sub_DC1960(v106, (__int64)v48, (__int64)v43, v3, 0);
    v122 = v124;
    v121 = v64;
    v123 = 0x300000000LL;
    if ( (_DWORD)v119 )
      sub_D915C0((__int64)&v122, (__int64)&v118, (unsigned int)v119, v65, v66, v67);
    v117 = v3;
    v68 = v106 + 1192;
    v116 = a3;
    v69 = (unsigned __int8)sub_D9E620(v106 + 1192, &v116, &v114) == 0;
    v73 = v114;
    if ( v69 )
    {
      v115 = v114;
      v74 = *(_DWORD *)(v106 + 1208);
      v75 = *(_DWORD *)(v106 + 1216);
      ++*(_QWORD *)(v106 + 1192);
      v76 = v74 + 1;
      v71 = 2 * v75;
      if ( 4 * (v74 + 1) >= 3 * v75 )
      {
        v75 *= 2;
      }
      else
      {
        v70 = v75 - *(_DWORD *)(v106 + 1212) - v76;
        if ( (unsigned int)v70 > v75 >> 3 )
        {
LABEL_79:
          *(_DWORD *)(v106 + 1208) = v76;
          if ( *v73 != -4096 || v73[1] != -4096 )
            --*(_DWORD *)(v106 + 1212);
          v77 = v116;
          v73[2] = 0;
          v73[4] = 0x300000000LL;
          *v73 = v77;
          v73[1] = v117;
          v73[3] = (__int64)(v73 + 5);
          goto LABEL_82;
        }
      }
      sub_DA6830(v68, v75);
      sub_D9E620(v68, &v116, &v115);
      v76 = *(_DWORD *)(v106 + 1208) + 1;
      v73 = v115;
      goto LABEL_79;
    }
LABEL_82:
    v78 = (__int64)v121;
    v79 = v73 + 2;
    a2 = (__int64)&v122;
    *v79 = v121;
    sub_D915C0((__int64)(v79 + 1), (__int64)&v122, v78, v70, v71, v72);
    *(_QWORD *)a1 = v121;
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_QWORD *)(a1 + 16) = 0x300000000LL;
    if ( (_DWORD)v123 )
    {
      a2 = (__int64)&v122;
      sub_D91460(a1 + 8, &v122, v80, v81, v82, v83);
    }
    v84 = v122;
    *(_BYTE *)(a1 + 48) = 1;
    if ( v84 != v124 )
      _libc_free(v84, &v122);
LABEL_40:
    if ( v125 != v127 )
      _libc_free(v125, a2);
    v44 = v118;
  }
LABEL_48:
  if ( v44 != v120 )
    _libc_free(v44, a2);
  return a1;
}
