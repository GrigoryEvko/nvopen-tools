// Function: sub_15AEA70
// Address: 0x15aea70
//
__int64 __fastcall sub_15AEA70(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  __int64 i; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdi
  _QWORD *v12; // rbx
  _QWORD *v13; // rdi
  __int64 *j; // rbx
  __int64 *v15; // rdi
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rax
  __int64 *v22; // r12
  __int64 v23; // rax
  unsigned __int64 *v24; // rbx
  unsigned int v25; // edx
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r13
  unsigned int v30; // ecx
  unsigned __int64 *v31; // rdx
  unsigned __int64 v32; // r8
  _BYTE *v33; // rcx
  unsigned int v34; // esi
  unsigned __int64 *v35; // rdx
  unsigned __int64 v36; // r9
  _BYTE *v37; // rbx
  unsigned int v38; // eax
  unsigned __int64 v39; // rsi
  __int64 v40; // r12
  unsigned int v41; // ebx
  unsigned __int64 v42; // rax
  _BYTE *v43; // r13
  __int64 v44; // rsi
  unsigned __int64 *v45; // rdx
  unsigned __int64 *v46; // rax
  __int64 v47; // rax
  unsigned int v48; // edi
  unsigned __int64 *v49; // rsi
  unsigned __int64 v50; // r10
  _BYTE *v51; // r13
  unsigned int v52; // edi
  unsigned __int64 *v53; // rsi
  _BYTE *v54; // r10
  _BYTE *v55; // r8
  unsigned int v56; // eax
  __int64 v57; // rax
  _QWORD *v58; // r12
  int v59; // r15d
  unsigned int v60; // ebx
  unsigned __int8 *v61; // r9
  __int64 v62; // rax
  __int64 v63; // rax
  unsigned __int8 *v64; // r13
  unsigned __int64 *v65; // rax
  unsigned __int8 *v66; // r10
  unsigned __int8 *v67; // rax
  int v68; // esi
  int v69; // esi
  int v70; // ecx
  int v71; // r11d
  int v72; // edx
  int v73; // edx
  int v74; // r9d
  int v75; // r10d
  int v76; // eax
  unsigned __int64 **v78; // rbx
  unsigned __int64 **v79; // r15
  int v80; // eax
  int v81; // r8d
  __int64 v82; // rax
  int v83; // r9d
  __int64 *v84; // [rsp+8h] [rbp-148h]
  __int64 *v85; // [rsp+10h] [rbp-140h]
  _QWORD *v86; // [rsp+18h] [rbp-138h]
  __int64 *v88; // [rsp+28h] [rbp-128h]
  __int16 *v89; // [rsp+40h] [rbp-110h]
  unsigned __int64 **v90; // [rsp+48h] [rbp-108h]
  unsigned __int8 v91; // [rsp+57h] [rbp-F9h]
  unsigned __int64 *v92; // [rsp+58h] [rbp-F8h]
  unsigned __int64 *v93; // [rsp+58h] [rbp-F8h]
  __int64 v94; // [rsp+58h] [rbp-F8h]
  _BYTE *v95; // [rsp+58h] [rbp-F8h]
  _BYTE *v96; // [rsp+60h] [rbp-F0h]
  unsigned __int64 *v97; // [rsp+60h] [rbp-F0h]
  unsigned __int64 *v98; // [rsp+60h] [rbp-F0h]
  __int64 v99; // [rsp+60h] [rbp-F0h]
  _BYTE *v100; // [rsp+60h] [rbp-F0h]
  unsigned int v101; // [rsp+60h] [rbp-F0h]
  _BYTE *v102; // [rsp+68h] [rbp-E8h]
  unsigned int v103; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v104; // [rsp+68h] [rbp-E8h]
  unsigned __int8 *v105; // [rsp+68h] [rbp-E8h]
  unsigned __int64 *v106; // [rsp+70h] [rbp-E0h] BYREF
  unsigned __int64 *v107; // [rsp+78h] [rbp-D8h] BYREF
  __int64 v108; // [rsp+80h] [rbp-D0h] BYREF
  unsigned __int64 *v109; // [rsp+88h] [rbp-C8h]
  __int64 v110; // [rsp+90h] [rbp-C0h]
  __int64 v111; // [rsp+98h] [rbp-B8h]
  __int64 v112; // [rsp+A0h] [rbp-B0h]
  __int64 v113; // [rsp+A8h] [rbp-A8h]
  __int64 v114; // [rsp+B0h] [rbp-A0h]
  __int64 v115; // [rsp+B8h] [rbp-98h]
  int v116; // [rsp+C0h] [rbp-90h]
  unsigned __int64 **v117; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v118; // [rsp+D8h] [rbp-78h]
  _BYTE v119[112]; // [rsp+E0h] [rbp-70h] BYREF

  v1 = sub_16321A0(a1, "llvm.dbg.declare", 16);
  v91 = 0;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 8);
    for ( i = v1; v2; v2 = *(_QWORD *)(i + 8) )
    {
      v4 = sub_1648700(v2);
      sub_15F20C0(v4, "llvm.dbg.declare", v5, v6);
    }
    sub_15E3D00(i);
    v91 = 1;
  }
  v7 = sub_16321A0(a1, "llvm.dbg.value", 14);
  if ( v7 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 8);
      if ( !v11 )
        break;
      v8 = sub_1648700(v11);
      sub_15F20C0(v8, "llvm.dbg.value", v9, v10);
    }
    sub_15E3D00(v7);
    v91 = 1;
  }
  v12 = (_QWORD *)a1[10];
  v86 = a1 + 9;
  while ( v86 != v12 )
  {
    v13 = v12;
    v12 = (_QWORD *)v12[1];
    sub_161F640(v13);
  }
  for ( j = (__int64 *)a1[2]; a1 + 1 != j; j = (__int64 *)j[1] )
  {
    v15 = j - 7;
    if ( !j )
      v15 = 0;
    sub_1626EF0(v15, 0);
  }
  v108 = 0;
  v16 = *a1;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v17 = sub_1627350(v16, 0, 0, 0, 1);
  v18 = 0;
  v113 = 0;
  v112 = sub_15BEF40(v16, 0, 0, v17, 0, 1);
  v21 = (__int64 *)a1[4];
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v84 = v21;
  if ( a1 + 3 != v21 )
  {
    while ( 1 )
    {
      v22 = v84 - 7;
      if ( !v84 )
        v22 = 0;
      v23 = sub_1626D20(v22);
      v24 = (unsigned __int64 *)v23;
      if ( v23 )
        break;
LABEL_26:
      v85 = v22 + 9;
      v88 = (__int64 *)v22[10];
      if ( v22 + 9 != v88 )
      {
        while ( 1 )
        {
          if ( !v88 )
            BUG();
          v89 = (__int16 *)v88[3];
          if ( v88 + 2 != (__int64 *)v89 )
            break;
LABEL_91:
          v88 = (__int64 *)v88[1];
          if ( v85 == v88 )
            goto LABEL_92;
        }
        while ( 2 )
        {
          if ( !v89 )
            BUG();
          v18 = *((_QWORD *)v89 + 3);
          if ( v18 )
          {
            v107 = (unsigned __int64 *)*((_QWORD *)v89 + 3);
            sub_1623A60(&v107, v18, 2);
            v28 = sub_15C70D0(&v107);
            v29 = sub_15C70F0(&v107);
            if ( v28 )
            {
              sub_15AE530((__int64)&v108, v28);
              if ( (_DWORD)v111 )
              {
                v30 = (v111 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                v31 = &v109[2 * v30];
                v32 = *v31;
                if ( *v31 == v28 )
                {
LABEL_34:
                  if ( v31 != &v109[2 * (unsigned int)v111] )
                  {
                    v33 = (_BYTE *)v31[1];
                    if ( v33 )
                      goto LABEL_36;
                    goto LABEL_107;
                  }
                }
                else
                {
                  v73 = 1;
                  while ( v32 != -4 )
                  {
                    v74 = v73 + 1;
                    v30 = (v111 - 1) & (v30 + v73);
                    v31 = &v109[2 * v30];
                    v32 = *v31;
                    if ( v28 == *v31 )
                      goto LABEL_34;
                    v73 = v74;
                  }
                }
              }
              v33 = (_BYTE *)v28;
LABEL_36:
              if ( (unsigned __int8)(*v33 - 4) > 0x1Eu )
LABEL_107:
                v33 = 0;
              v91 |= v28 != (_QWORD)v33;
            }
            else
            {
              v33 = 0;
            }
            if ( v29 )
            {
              v102 = v33;
              sub_15AE530((__int64)&v108, v29);
              v33 = v102;
              if ( (_DWORD)v111 )
              {
                v34 = (v111 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v35 = &v109[2 * v34];
                v36 = *v35;
                if ( v29 == *v35 )
                {
LABEL_41:
                  if ( v35 != &v109[2 * (unsigned int)v111] )
                  {
                    v37 = (_BYTE *)v35[1];
                    if ( v37 )
                      goto LABEL_43;
                    goto LABEL_106;
                  }
                }
                else
                {
                  v72 = 1;
                  while ( v36 != -4 )
                  {
                    v75 = v72 + 1;
                    v34 = (v111 - 1) & (v34 + v72);
                    v35 = &v109[2 * v34];
                    v36 = *v35;
                    if ( v29 == *v35 )
                      goto LABEL_41;
                    v72 = v75;
                  }
                }
              }
              v37 = (_BYTE *)v29;
LABEL_43:
              if ( (unsigned __int8)(*v37 - 4) > 0x1Eu )
LABEL_106:
                v37 = 0;
              v91 |= v29 != (_QWORD)v37;
            }
            else
            {
              v37 = 0;
            }
            v96 = v33;
            v103 = sub_15C70C0(&v107);
            v38 = sub_15C70B0(&v107);
            sub_15C7110(&v117, v38, v103, v96, v37);
            if ( v89 + 12 == (__int16 *)&v117 )
            {
              if ( v117 )
                sub_161E7C0(v89 + 12);
            }
            else
            {
              if ( *((_QWORD *)v89 + 3) )
                sub_161E7C0(v89 + 12);
              v39 = (unsigned __int64)v117;
              *((_QWORD *)v89 + 3) = v117;
              if ( v39 )
                sub_1623210(&v117, v39, v89 + 12);
            }
            v18 = (unsigned __int64)v107;
            if ( v107 )
              sub_161E7C0(&v107);
          }
          v117 = (unsigned __int64 **)v119;
          v118 = 0x200000000LL;
          if ( *((_QWORD *)v89 + 3) || *(v89 - 3) < 0 )
          {
            v18 = (unsigned __int64)&v117;
            sub_161F840(v89 - 12, &v117);
            v104 = (unsigned __int64)v117;
            v90 = &v117[2 * (unsigned int)v118];
            if ( v117 != v90 )
            {
              while ( 1 )
              {
                v40 = *(_QWORD *)(v104 + 8);
                if ( v40 )
                {
                  if ( *(_BYTE *)v40 == 4 )
                  {
                    v19 = *(unsigned int *)(v40 + 8);
                    if ( (_DWORD)v19 )
                      break;
                  }
                }
LABEL_86:
                v104 += 16LL;
                if ( v90 == (unsigned __int64 **)v104 )
                {
                  v90 = v117;
                  goto LABEL_88;
                }
              }
              v41 = 0;
              while ( 2 )
              {
                v20 = (unsigned int)v19;
                v42 = v41 - (unsigned __int64)(unsigned int)v19;
                v43 = *(_BYTE **)(v40 + 8 * v42);
                if ( !v43 || *v43 != 5 )
                  goto LABEL_85;
                v44 = *(_QWORD *)(v40 + 8 * v42);
                v107 = 0;
                sub_15C7080(&v106, v44);
                v45 = v107;
                v46 = v106;
                v18 = (unsigned __int64)v107;
                if ( v106 )
                {
                  v92 = v107;
                  v97 = v106;
                  sub_161E7C0(&v106);
                  v18 = (unsigned __int64)v107;
                  v45 = v92;
                  v46 = v97;
                }
                if ( v18 )
                {
                  v93 = v45;
                  v98 = v46;
                  sub_161E7C0(&v107);
                  v45 = v93;
                  v46 = v98;
                }
                if ( v46 == v45 )
                  goto LABEL_84;
                sub_15C7080(&v106, v43);
                v99 = sub_15C70D0(&v106);
                v47 = sub_15C70F0(&v106);
                if ( v99 )
                {
                  v94 = v47;
                  sub_15AE530((__int64)&v108, v99);
                  v47 = v94;
                  if ( (_DWORD)v111 )
                  {
                    v48 = (v111 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
                    v49 = &v109[2 * v48];
                    v50 = *v49;
                    if ( v99 == *v49 )
                    {
LABEL_68:
                      if ( v49 != &v109[2 * (unsigned int)v111] )
                      {
                        v51 = (_BYTE *)v49[1];
                        if ( v51 )
                          goto LABEL_70;
                        goto LABEL_71;
                      }
                    }
                    else
                    {
                      v68 = 1;
                      while ( v50 != -4 )
                      {
                        v71 = v68 + 1;
                        v48 = (v111 - 1) & (v68 + v48);
                        v49 = &v109[2 * v48];
                        v50 = *v49;
                        if ( v99 == *v49 )
                          goto LABEL_68;
                        v68 = v71;
                      }
                    }
                  }
                  v51 = (_BYTE *)v99;
LABEL_70:
                  if ( (unsigned __int8)(*v51 - 4) > 0x1Eu )
LABEL_71:
                    v51 = 0;
                  v91 |= v99 != (_QWORD)v51;
                  if ( v94 )
                  {
LABEL_73:
                    v100 = (_BYTE *)v47;
                    sub_15AE530((__int64)&v108, v47);
                    if ( (_DWORD)v111 )
                    {
                      v52 = (v111 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
                      v53 = &v109[2 * v52];
                      v54 = (_BYTE *)*v53;
                      if ( v100 == (_BYTE *)*v53 )
                      {
LABEL_75:
                        if ( v53 != &v109[2 * (unsigned int)v111] )
                        {
                          v55 = (_BYTE *)v53[1];
                          if ( v55 )
                            goto LABEL_77;
                          goto LABEL_78;
                        }
                      }
                      else
                      {
                        v69 = 1;
                        while ( v54 != (_BYTE *)-4LL )
                        {
                          v70 = v69 + 1;
                          v52 = (v111 - 1) & (v69 + v52);
                          v53 = &v109[2 * v52];
                          v54 = (_BYTE *)*v53;
                          if ( v100 == (_BYTE *)*v53 )
                            goto LABEL_75;
                          v69 = v70;
                        }
                      }
                    }
                    v55 = v100;
LABEL_77:
                    if ( (unsigned __int8)(*v55 - 4) > 0x1Eu )
LABEL_78:
                      v55 = 0;
                    v91 |= v100 != v55;
LABEL_80:
                    v95 = v55;
                    v101 = sub_15C70C0(&v106);
                    v56 = sub_15C70B0(&v106);
                    sub_15C7110(&v107, v56, v101, v51, v95);
                    v57 = sub_15C70A0(&v107);
                    sub_1630830(v40, v41, v57);
                    if ( v107 )
                      sub_161E7C0(&v107);
                    v18 = (unsigned __int64)v106;
                    if ( v106 )
                      sub_161E7C0(&v106);
LABEL_84:
                    v19 = *(unsigned int *)(v40 + 8);
LABEL_85:
                    if ( ++v41 >= (unsigned int)v19 )
                      goto LABEL_86;
                    continue;
                  }
                }
                else
                {
                  v51 = 0;
                  if ( v47 )
                    goto LABEL_73;
                }
                break;
              }
              v55 = 0;
              goto LABEL_80;
            }
LABEL_88:
            if ( v90 != (unsigned __int64 **)v119 )
              _libc_free((unsigned __int64)v90);
          }
          v89 = (__int16 *)*((_QWORD *)v89 + 1);
          if ( v88 + 2 == (__int64 *)v89 )
            goto LABEL_91;
          continue;
        }
      }
LABEL_92:
      v84 = (__int64 *)v84[1];
      if ( a1 + 3 == v84 )
        goto LABEL_93;
    }
    sub_15AE530((__int64)&v108, v23);
    if ( (_DWORD)v111 )
    {
      v25 = (v111 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v26 = &v109[2 * v25];
      v27 = (unsigned __int64 *)*v26;
      if ( v24 == (unsigned __int64 *)*v26 )
      {
LABEL_22:
        if ( v26 != &v109[2 * (unsigned int)v111] )
        {
          v18 = v26[1];
          if ( !v18 )
            goto LABEL_108;
          goto LABEL_24;
        }
      }
      else
      {
        v80 = 1;
        while ( v27 != (unsigned __int64 *)-4LL )
        {
          v83 = v80 + 1;
          v25 = (v111 - 1) & (v80 + v25);
          v26 = &v109[2 * v25];
          v27 = (unsigned __int64 *)*v26;
          if ( v24 == (unsigned __int64 *)*v26 )
            goto LABEL_22;
          v80 = v83;
        }
      }
    }
    v18 = (unsigned __int64)v24;
LABEL_24:
    if ( (unsigned __int8)(*(_BYTE *)v18 - 4) <= 0x1Eu )
    {
LABEL_25:
      v91 |= v24 != (unsigned __int64 *)v18;
      sub_1627150(v22, v18);
      goto LABEL_26;
    }
LABEL_108:
    v18 = 0;
    goto LABEL_25;
  }
LABEL_93:
  v58 = (_QWORD *)a1[10];
  if ( v86 != v58 )
  {
    while ( 1 )
    {
      v117 = (unsigned __int64 **)v119;
      v118 = 0x800000000LL;
      v59 = sub_161F520(v58, v18, v19, v20);
      if ( v59 )
        break;
LABEL_142:
      if ( v91 )
      {
        sub_161F550(v58);
        v78 = v117;
        v79 = &v117[(unsigned int)v118];
        if ( v117 != v79 )
        {
          do
          {
            v18 = (unsigned __int64)*v78;
            if ( *v78 )
              sub_1623CA0(v58, v18);
            ++v78;
          }
          while ( v79 != v78 );
          v79 = v117;
        }
        if ( v79 != (unsigned __int64 **)v119 )
          _libc_free((unsigned __int64)v79);
      }
      else if ( v117 != (unsigned __int64 **)v119 )
      {
        _libc_free((unsigned __int64)v117);
      }
      v58 = (_QWORD *)v58[1];
      if ( v86 == v58 )
        goto LABEL_146;
    }
    v60 = 0;
    while ( 1 )
    {
      v18 = v60;
      v63 = sub_161F530(v58, v60);
      v64 = (unsigned __int8 *)v63;
      if ( v63 )
        break;
      v62 = (unsigned int)v118;
      v61 = 0;
      if ( (unsigned int)v118 >= HIDWORD(v118) )
      {
LABEL_141:
        v18 = (unsigned __int64)v119;
        v105 = v61;
        sub_16CD150(&v117, v119, 0, 8);
        v62 = (unsigned int)v118;
        v61 = v105;
      }
LABEL_98:
      v19 = (unsigned __int64)v117;
      ++v60;
      v117[v62] = (unsigned __int64 *)v61;
      LODWORD(v118) = v118 + 1;
      if ( v59 == v60 )
        goto LABEL_142;
    }
    v18 = v63;
    sub_15AE530((__int64)&v108, v63);
    if ( (_DWORD)v111 )
    {
      v18 = (unsigned __int64)v109;
      v20 = ((_DWORD)v111 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
      v65 = &v109[2 * v20];
      v66 = (unsigned __int8 *)*v65;
      if ( v64 == (unsigned __int8 *)*v65 )
      {
LABEL_102:
        if ( v65 != &v109[2 * (unsigned int)v111] )
        {
          v67 = (unsigned __int8 *)v65[1];
          if ( !v67 )
            goto LABEL_105;
          goto LABEL_104;
        }
      }
      else
      {
        v76 = 1;
        while ( v66 != (unsigned __int8 *)-4LL )
        {
          v81 = v76 + 1;
          v82 = ((_DWORD)v111 - 1) & (unsigned int)(v20 + v76);
          v20 = (unsigned int)v82;
          v65 = &v109[2 * v82];
          v66 = (unsigned __int8 *)*v65;
          if ( v64 == (unsigned __int8 *)*v65 )
            goto LABEL_102;
          v76 = v81;
        }
      }
    }
    v67 = v64;
LABEL_104:
    v20 = *v67;
    if ( (unsigned __int8)(v20 - 4) <= 0x1Eu )
    {
      v61 = v67;
      goto LABEL_97;
    }
LABEL_105:
    v61 = 0;
    v67 = 0;
LABEL_97:
    v91 |= v64 != v67;
    v62 = (unsigned int)v118;
    if ( (unsigned int)v118 >= HIDWORD(v118) )
      goto LABEL_141;
    goto LABEL_98;
  }
LABEL_146:
  j___libc_free_0(v114);
  j___libc_free_0(v109);
  return v91;
}
