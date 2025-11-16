// Function: sub_F40FD0
// Address: 0xf40fd0
//
__int64 __fastcall sub_F40FD0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rbx
  __int64 v10; // rsi
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r8
  int v15; // esi
  int v16; // esi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r10
  __int64 v20; // r14
  __int64 v21; // r12
  unsigned __int8 *v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // rdi
  int v26; // esi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r11
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 *v32; // r14
  __int64 v33; // rax
  __int64 *v34; // r12
  __int64 *v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned __int64 v40; // rdx
  __int64 v41; // rcx
  unsigned __int64 v42; // rdx
  __int64 v43; // rcx
  unsigned __int64 v44; // rdx
  __int64 v45; // r14
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // r12
  unsigned __int8 *v49; // rax
  _QWORD *v50; // r13
  _QWORD *v51; // rax
  _QWORD *v52; // rbx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r13
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  _QWORD *v63; // rbx
  _QWORD *v64; // r15
  void (__fastcall *v65)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v66; // rax
  int v67; // edx
  __int64 v68; // rdx
  unsigned int v69; // ecx
  __int64 *v70; // rax
  __int64 v71; // r8
  __int64 *v72; // rbx
  __int64 v73; // rcx
  __int64 *v74; // rax
  __int64 v75; // rdi
  __int64 *v76; // rdi
  __int64 *v77; // rax
  int v79; // edx
  __int64 v80; // rdx
  __int64 v81; // r13
  unsigned __int16 v82; // bx
  _QWORD *v83; // rax
  __int64 v84; // r15
  unsigned __int16 v85; // bx
  _QWORD *v86; // rdi
  unsigned __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rdx
  int v90; // edx
  __int64 *v91; // rax
  int v92; // eax
  int v93; // r8d
  int v94; // edx
  int v95; // eax
  int v96; // r9d
  __int64 v97; // [rsp+0h] [rbp-3A0h]
  __int64 v99; // [rsp+8h] [rbp-398h]
  __int64 v100; // [rsp+8h] [rbp-398h]
  unsigned __int16 v102; // [rsp+18h] [rbp-388h]
  __int64 *v103; // [rsp+18h] [rbp-388h]
  __int64 v104; // [rsp+18h] [rbp-388h]
  __int64 v106; // [rsp+28h] [rbp-378h]
  __int64 v108; // [rsp+38h] [rbp-368h] BYREF
  __int64 **v109; // [rsp+40h] [rbp-360h] BYREF
  __int64 v110; // [rsp+48h] [rbp-358h]
  _BYTE v111[32]; // [rsp+50h] [rbp-350h] BYREF
  unsigned __int64 *v112; // [rsp+70h] [rbp-330h] BYREF
  __int64 v113; // [rsp+78h] [rbp-328h]
  _BYTE v114[48]; // [rsp+80h] [rbp-320h] BYREF
  _BYTE *v115; // [rsp+B0h] [rbp-2F0h] BYREF
  __int64 v116; // [rsp+B8h] [rbp-2E8h]
  _BYTE v117[512]; // [rsp+C0h] [rbp-2E0h] BYREF
  __int64 v118; // [rsp+2C0h] [rbp-E0h]
  __int64 v119; // [rsp+2C8h] [rbp-D8h]
  __int64 v120; // [rsp+2D0h] [rbp-D0h]
  __int64 v121; // [rsp+2D8h] [rbp-C8h]
  char v122; // [rsp+2E0h] [rbp-C0h]
  __int64 v123; // [rsp+2E8h] [rbp-B8h]
  char *v124; // [rsp+2F0h] [rbp-B0h]
  __int64 v125; // [rsp+2F8h] [rbp-A8h]
  int v126; // [rsp+300h] [rbp-A0h]
  char v127; // [rsp+304h] [rbp-9Ch]
  char v128; // [rsp+308h] [rbp-98h] BYREF
  __int16 v129; // [rsp+348h] [rbp-58h]
  _QWORD *v130; // [rsp+350h] [rbp-50h]
  _QWORD *v131; // [rsp+358h] [rbp-48h]
  __int64 v132; // [rsp+360h] [rbp-40h]

  v108 = a1;
  v7 = sub_AA4FF0(a2);
  v9 = v7;
  if ( v7 )
  {
    if ( !a4 )
    {
      v87 = (unsigned int)*(unsigned __int8 *)(v7 - 24) - 39;
      if ( (unsigned int)v87 > 0x38 )
        return sub_F41C30(v108, a2, *a5, a5[2], a5[3], a6);
      v88 = 0x100060000000001LL;
      if ( !_bittest64(&v88, v87) )
        return sub_F41C30(v108, a2, *a5, a5[2], a5[3], a6);
    }
  }
  else if ( !a4 )
  {
    BUG();
  }
  v10 = a5[2];
  v11 = *((_BYTE *)a5 + 36) == 0;
  v109 = (__int64 **)v111;
  v110 = 0x400000000LL;
  v12 = v108;
  v106 = v10;
  v13 = v108;
  if ( v11 )
    goto LABEL_44;
  if ( !v10 )
    goto LABEL_44;
  v14 = *(_QWORD *)(v10 + 8);
  v15 = *(_DWORD *)(v10 + 24);
  if ( !v15 )
    goto LABEL_44;
  v16 = v15 - 1;
  v17 = v16 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
  v18 = (__int64 *)(v14 + 16LL * v17);
  v19 = *v18;
  if ( *v18 != v108 )
  {
    v94 = 1;
    while ( v19 != -4096 )
    {
      v8 = (unsigned int)(v94 + 1);
      v17 = v16 & (v94 + v17);
      v18 = (__int64 *)(v14 + 16LL * v17);
      v19 = *v18;
      if ( *v18 == v108 )
        goto LABEL_7;
      v94 = v8;
    }
    goto LABEL_44;
  }
LABEL_7:
  v20 = v18[1];
  if ( !v20 )
    goto LABEL_44;
  v21 = *(_QWORD *)(a2 + 16);
  if ( !v21 )
    goto LABEL_112;
  while ( 1 )
  {
    v22 = *(unsigned __int8 **)(v21 + 24);
    v23 = *v22;
    if ( (unsigned __int8)(v23 - 30) <= 0xAu )
      break;
    v21 = *(_QWORD *)(v21 + 8);
    if ( !v21 )
      goto LABEL_112;
  }
LABEL_13:
  v24 = *((_QWORD *)v22 + 5);
  if ( v24 == v12 )
    goto LABEL_20;
  v25 = *(_QWORD *)(v106 + 8);
  v26 = *(_DWORD *)(v106 + 24);
  if ( !v26 )
    goto LABEL_111;
  v23 = (unsigned int)(v26 - 1);
  v27 = v23 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
  v28 = (__int64 *)(v25 + 16LL * v27);
  v29 = *v28;
  if ( v24 != *v28 )
  {
    v90 = 1;
    while ( v29 != -4096 )
    {
      v8 = (unsigned int)(v90 + 1);
      v27 = v23 & (v90 + v27);
      v28 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v28;
      if ( v24 == *v28 )
        goto LABEL_16;
      v90 = v8;
    }
    goto LABEL_111;
  }
LABEL_16:
  if ( v20 != v28[1] )
  {
LABEL_111:
    LODWORD(v110) = 0;
    goto LABEL_112;
  }
  v30 = (unsigned int)v110;
  v31 = (unsigned int)v110 + 1LL;
  if ( v31 > HIDWORD(v110) )
  {
    v23 = (__int64)v111;
    v97 = v24;
    sub_C8D5F0((__int64)&v109, v111, v31, 8u, v24, v8);
    v30 = (unsigned int)v110;
    v24 = v97;
  }
  v109[v30] = (__int64 *)v24;
  LODWORD(v110) = v110 + 1;
LABEL_20:
  while ( 1 )
  {
    v21 = *(_QWORD *)(v21 + 8);
    if ( !v21 )
      break;
    v22 = *(unsigned __int8 **)(v21 + 24);
    if ( (unsigned __int8)(*v22 - 30) <= 0xAu )
    {
      v12 = v108;
      goto LABEL_13;
    }
  }
  v32 = (__int64 *)v109;
  v33 = 8LL * (unsigned int)v110;
  v34 = (__int64 *)v109;
  v35 = (__int64 *)&v109[(unsigned __int64)v33 / 8];
  v36 = v33 >> 3;
  v37 = v33 >> 5;
  if ( v37 )
  {
    while ( 1 )
    {
      v38 = *(_QWORD *)(*v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v38 == *v34 + 48 )
        goto LABEL_150;
      if ( !v38 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v38 - 24) - 30 > 0xA )
LABEL_150:
        BUG();
      if ( *(_BYTE *)(v38 - 24) == 33 )
        goto LABEL_107;
      v39 = v34[1];
      v40 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v40 == v39 + 48 )
        goto LABEL_156;
      if ( !v40 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 > 0xA )
LABEL_156:
        BUG();
      if ( *(_BYTE *)(v40 - 24) == 33 )
      {
        ++v34;
        goto LABEL_107;
      }
      v41 = v34[2];
      v42 = *(_QWORD *)(v41 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v42 == v41 + 48 )
        goto LABEL_148;
      if ( !v42 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
LABEL_148:
        BUG();
      if ( *(_BYTE *)(v42 - 24) == 33 )
      {
        v34 += 2;
        goto LABEL_107;
      }
      v43 = v34[3];
      v44 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v44 == v43 + 48 )
        goto LABEL_152;
      if ( !v44 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v44 - 24) - 30 > 0xA )
LABEL_152:
        BUG();
      if ( *(_BYTE *)(v44 - 24) == 33 )
      {
        v34 += 3;
        goto LABEL_107;
      }
      v34 += 4;
      if ( !--v37 )
      {
        v36 = v35 - v34;
        break;
      }
    }
  }
  if ( v36 == 2 )
    goto LABEL_41;
  if ( v36 != 3 )
  {
    v12 = v108;
    if ( v36 == 1 )
    {
LABEL_43:
      v13 = v108;
      if ( *(_BYTE *)sub_986580(*v34) != 33 )
        goto LABEL_44;
      goto LABEL_107;
    }
LABEL_112:
    v13 = v12;
    goto LABEL_44;
  }
  if ( *(_BYTE *)sub_986580(*v34) == 33 )
    goto LABEL_107;
  ++v34;
LABEL_41:
  if ( *(_BYTE *)sub_986580(*v34) != 33 )
  {
    ++v34;
    goto LABEL_43;
  }
LABEL_107:
  if ( v35 != v34 )
  {
    v48 = 0;
    goto LABEL_84;
  }
  v13 = v108;
LABEL_44:
  v45 = *(_QWORD *)(v13 + 72);
  v46 = sub_AA48A0(v13);
  v47 = sub_22077B0(80);
  v48 = v47;
  if ( v47 )
    sub_AA4D50(v47, v46, a6, v45, a2);
  v49 = (unsigned __int8 *)sub_986580(v108);
  sub_F34950(v49, v48);
  sub_F34A80(a2, v108, v48, a4);
  if ( !a4 )
  {
    if ( !v9 )
      BUG();
    v79 = *(unsigned __int8 *)(v9 - 24);
    if ( (unsigned int)(v79 - 80) > 1 )
    {
      if ( (_BYTE)v79 == 39 )
      {
        v80 = **(_QWORD **)(v9 - 32);
        goto LABEL_90;
      }
      if ( (_BYTE)v79 != 80 )
      {
        if ( (_BYTE)v79 != 95 )
          BUG();
        v80 = *(_QWORD *)(v9 + 16);
        goto LABEL_90;
      }
    }
    v80 = *(_QWORD *)(v9 - 56);
LABEL_90:
    v100 = v80;
    sub_B43C20((__int64)&v115, v48);
    v81 = (__int64)v115;
    v82 = v116;
    v83 = sub_BD2C40(72, 1u);
    v84 = (__int64)v83;
    if ( v83 )
      sub_B4C840((__int64)v83, 51, v100, 0, 0, 1u, a6, v81, v82);
    sub_B43C20((__int64)&v115, v48);
    v85 = v116;
    v23 = 2 - (unsigned int)(a2 == 0);
    v104 = (__int64)v115;
    v86 = sub_BD2C40(72, v23);
    if ( v86 )
    {
      v23 = v84;
      sub_B4BF70((__int64)v86, v84, a2, (2 - (a2 == 0)) & 0x1FFFFFFF, v104, v85);
    }
    goto LABEL_50;
  }
  v50 = (_QWORD *)sub_B47F80(a3);
  sub_B43C20((__int64)&v115, v48);
  v99 = (__int64)v115;
  v102 = v116;
  v51 = sub_BD2C40(72, 1u);
  v52 = v51;
  if ( v51 )
    sub_B4C8F0((__int64)v51, a2, 1u, v99, v102);
  sub_B44220(v50, (__int64)(v52 + 3), 0);
  v23 = (__int64)v50;
  sub_F0A850(a4, (__int64)v50, v48);
LABEL_50:
  v56 = *a5;
  if ( *a5 | v106 )
  {
    v103 = (__int64 *)a5[3];
    if ( v56 )
    {
      v115 = v117;
      v116 = 0x1000000000LL;
      v129 = 0;
      v124 = &v128;
      v113 = 0x300000000LL;
      v118 = 0;
      v119 = 0;
      v120 = v56;
      v121 = 0;
      v122 = 1;
      v123 = 0;
      v125 = 8;
      v126 = 0;
      v127 = 1;
      v130 = 0;
      v131 = 0;
      v132 = 0;
      v112 = (unsigned __int64 *)v114;
      sub_F35FA0((__int64)&v112, v108, v48 & 0xFFFFFFFFFFFFFFFBLL, v53, v54, v55);
      sub_F35FA0((__int64)&v112, v48, a2 & 0xFFFFFFFFFFFFFFFBLL, v57, v58, v59);
      sub_F35FA0((__int64)&v112, v108, a2 & 0xFFFFFFFFFFFFFFFBLL | 4, v60, v61, v62);
      v23 = (__int64)v112;
      sub_FFB3D0(&v115, v112, (unsigned int)v113);
      sub_FFCE90(&v115);
      sub_FFD870(&v115);
      sub_FFBC40(&v115);
      if ( v103 )
      {
        v23 = (__int64)v112;
        sub_D75690(v103, v112, (unsigned int)v113, v56, 0);
        if ( byte_4F8F8E8[0] )
        {
          v23 = 0;
          nullsub_390(*v103, 0);
        }
      }
      if ( v112 != (unsigned __int64 *)v114 )
        _libc_free(v112, v23);
      sub_FFCE90(&v115);
      sub_FFD870(&v115);
      sub_FFBC40(&v115);
      v63 = v131;
      v64 = v130;
      if ( v131 != v130 )
      {
        do
        {
          v65 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v64[7];
          *v64 = &unk_49E5048;
          if ( v65 )
          {
            v23 = (__int64)(v64 + 5);
            v65(v64 + 5, v64 + 5, 3);
          }
          *v64 = &unk_49DB368;
          v66 = v64[3];
          if ( v66 != 0 && v66 != -4096 && v66 != -8192 )
            sub_BD60C0(v64 + 1);
          v64 += 9;
        }
        while ( v63 != v64 );
        v64 = v130;
      }
      if ( v64 )
      {
        v23 = v132 - (_QWORD)v64;
        j_j___libc_free_0(v64, v132 - (_QWORD)v64);
      }
      if ( !v127 )
        _libc_free(v124, v23);
      if ( v115 != v117 )
        _libc_free(v115, v23);
    }
    if ( v106 )
    {
      v67 = *(_DWORD *)(v106 + 24);
      v23 = *(_QWORD *)(v106 + 8);
      if ( v67 )
      {
        v68 = (unsigned int)(v67 - 1);
        v69 = v68 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
        v70 = (__int64 *)(v23 + 16LL * v69);
        v71 = *v70;
        if ( v108 == *v70 )
        {
LABEL_74:
          v72 = (__int64 *)v70[1];
          if ( v72 )
          {
            v73 = (unsigned int)v68 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v74 = (__int64 *)(v23 + 16 * v73);
            v75 = *v74;
            if ( a2 == *v74 )
            {
LABEL_76:
              v76 = (__int64 *)v74[1];
              if ( v76 )
              {
                v77 = (__int64 *)v74[1];
                if ( v72 != v76 )
                {
                  while ( 1 )
                  {
                    v77 = (__int64 *)*v77;
                    if ( v72 == v77 )
                      break;
                    if ( !v77 )
                    {
                      if ( v76 != v72 )
                      {
                        v91 = v72;
                        while ( 1 )
                        {
                          v91 = (__int64 *)*v91;
                          if ( v76 == v91 )
                            break;
                          if ( !v91 )
                          {
                            v76 = (__int64 *)*v76;
                            if ( v76 )
                              break;
                            goto LABEL_82;
                          }
                        }
                      }
                      sub_D4F330(v76, v48, v106);
                      goto LABEL_82;
                    }
                  }
                }
                sub_D4F330(v72, v48, v106);
              }
            }
            else
            {
              v92 = 1;
              while ( v75 != -4096 )
              {
                v93 = v92 + 1;
                v73 = (unsigned int)v68 & (v92 + (_DWORD)v73);
                v74 = (__int64 *)(v23 + 16LL * (unsigned int)v73);
                v75 = *v74;
                if ( a2 == *v74 )
                  goto LABEL_76;
                v92 = v93;
              }
            }
LABEL_82:
            v23 = a2;
            if ( !(unsigned __int8)sub_B19060((__int64)(v72 + 7), a2, v68, v73) )
            {
              if ( *((_BYTE *)a5 + 34) )
              {
                v23 = 1;
                sub_F34B50(&v108, 1, v48, a2);
              }
              v32 = (__int64 *)v109;
              if ( !(_DWORD)v110 )
                goto LABEL_84;
              v23 = (__int64)v109;
              v89 = sub_F40FB0(a2, v109, (unsigned int)v110, "split", v56, v106, v103, *((_BYTE *)a5 + 34));
              if ( *((_BYTE *)a5 + 34) )
              {
                v23 = (unsigned int)v110;
                sub_F34B50((__int64 *)v109, (unsigned int)v110, v89, a2);
              }
            }
          }
        }
        else
        {
          v95 = 1;
          while ( v71 != -4096 )
          {
            v96 = v95 + 1;
            v69 = v68 & (v95 + v69);
            v70 = (__int64 *)(v23 + 16LL * v69);
            v71 = *v70;
            if ( v108 == *v70 )
              goto LABEL_74;
            v95 = v96;
          }
        }
      }
    }
  }
  v32 = (__int64 *)v109;
LABEL_84:
  if ( v32 != (__int64 *)v111 )
    _libc_free(v32, v23);
  return v48;
}
