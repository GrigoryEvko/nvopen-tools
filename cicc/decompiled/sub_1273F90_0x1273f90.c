// Function: sub_1273F90
// Address: 0x1273f90
//
void __fastcall sub_1273F90(__int64 *a1)
{
  __int64 v1; // r13
  size_t v2; // rdx
  char *v3; // r12
  unsigned __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rsi
  __int64 v15; // r13
  __int64 v16; // rdx
  int v17; // eax
  size_t v18; // rax
  size_t v19; // r15
  _QWORD *v20; // rdx
  __int64 v21; // r15
  size_t v22; // rax
  __int64 v23; // rax
  _BYTE *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  _BYTE *v30; // rsi
  __int64 v31; // rax
  const char *v32; // r12
  int v33; // r15d
  _BYTE *v34; // rax
  size_t v35; // r8
  __int64 v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rax
  _BYTE *v43; // rsi
  __int64 v44; // rax
  _BYTE *v45; // rsi
  __int64 v46; // rax
  _BYTE *v47; // rsi
  __int64 v48; // rax
  _BYTE *v49; // rsi
  __int64 v50; // rax
  _BYTE *v51; // rsi
  _BYTE *v52; // rsi
  _QWORD *v53; // rdi
  size_t v54; // rax
  size_t v55; // r15
  _QWORD *v56; // rdx
  size_t v57; // rax
  __int64 v58; // rax
  _BYTE *v59; // rsi
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rax
  _BYTE *v66; // rax
  size_t v67; // r15
  __int64 v68; // rdx
  int v69; // r12d
  _BYTE *v70; // rsi
  __int64 v71; // rdi
  __int64 v72; // rax
  _BYTE *v73; // rsi
  _BYTE *v74; // rdx
  _BYTE *v75; // rsi
  _BYTE *v76; // rsi
  _QWORD *v77; // rdi
  _QWORD *v78; // rdi
  _BYTE *v79; // rax
  size_t v80; // r15
  __int64 v81; // rdx
  _QWORD *v82; // rdi
  __int64 v83; // [rsp+0h] [rbp-160h]
  __int64 v84; // [rsp+8h] [rbp-158h]
  __int64 v85; // [rsp+10h] [rbp-150h]
  bool v86; // [rsp+27h] [rbp-139h]
  __int64 v87; // [rsp+28h] [rbp-138h]
  _QWORD *v88; // [rsp+30h] [rbp-130h]
  int na; // [rsp+50h] [rbp-110h]
  size_t n; // [rsp+50h] [rbp-110h]
  size_t nb; // [rsp+50h] [rbp-110h]
  __int64 v93; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v94; // [rsp+70h] [rbp-F0h] BYREF
  _BYTE *v95; // [rsp+78h] [rbp-E8h]
  _BYTE *v96; // [rsp+80h] [rbp-E0h]
  _BYTE *v97; // [rsp+90h] [rbp-D0h] BYREF
  _BYTE *v98; // [rsp+98h] [rbp-C8h]
  _BYTE *v99; // [rsp+A0h] [rbp-C0h]
  _BYTE *v100; // [rsp+B0h] [rbp-B0h] BYREF
  _BYTE *v101; // [rsp+B8h] [rbp-A8h]
  _BYTE *v102; // [rsp+C0h] [rbp-A0h]
  _QWORD v103[2]; // [rsp+D0h] [rbp-90h] BYREF
  _QWORD v104[2]; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v105; // [rsp+F0h] [rbp-70h] BYREF
  size_t v106; // [rsp+F8h] [rbp-68h]
  _QWORD v107[2]; // [rsp+100h] [rbp-60h] BYREF
  __int64 v108; // [rsp+110h] [rbp-50h] BYREF
  _BYTE *v109; // [rsp+118h] [rbp-48h]
  _QWORD v110[8]; // [rsp+120h] [rbp-40h] BYREF

  if ( a1[56] == a1[57] )
    return;
  v1 = *a1;
  v2 = 0;
  v3 = off_4CD4988;
  if ( off_4CD4988 )
    v2 = strlen(off_4CD4988);
  v4 = (unsigned __int64)v3;
  v84 = sub_1632440(v1, v3, v2);
  v5 = a1[56];
  v6 = (a1[57] - v5) >> 3;
  if ( (int)v6 <= 0 )
    return;
  v7 = (unsigned int)(v6 - 1);
  v85 = 0;
  v83 = 8LL * (unsigned int)v7;
  while ( 2 )
  {
    v8 = *(_QWORD **)(v5 + v85);
    v96 = 0;
    v94 = 0;
    v9 = v8[4] - v8[3];
    v108 = 4;
    v10 = v8;
    v109 = 0;
    v11 = v8[2];
    v12 = v9 >> 4;
    v88 = v8;
    v110[0] = v11;
    v95 = 0;
    LOBYTE(v7) = v11 != -8;
    if ( ((v11 != 0) & (unsigned __int8)v7) != 0 && v11 != -16 )
    {
      v4 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      sub_1649AC0(&v108, v4);
      v11 = v110[0];
    }
    v13 = sub_1624210(v11, v4, v7, v10);
    v14 = v95;
    v105 = v13;
    if ( v95 == v96 )
    {
      sub_1273E00((__int64)&v94, v95, &v105);
    }
    else
    {
      if ( v95 )
      {
        *(_QWORD *)v95 = v13;
        v14 = v95;
      }
      v95 = v14 + 8;
    }
    if ( v110[0] != 0 && v110[0] != -8 && v110[0] != -16 )
      sub_1649B30(&v108);
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    if ( (int)v12 <= 0 )
    {
LABEL_129:
      v52 = v95;
      goto LABEL_67;
    }
    v15 = 0;
    v87 = 16 * ((unsigned int)(v12 - 1) + 1LL);
    do
    {
      while ( 1 )
      {
        v31 = v15 + v88[3];
        v32 = *(const char **)v31;
        v33 = *(_DWORD *)(v31 + 8);
        v108 = (__int64)v110;
        if ( !v32 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v34 = (_BYTE *)strlen(v32);
        v105 = (__int64)v34;
        v35 = (size_t)v34;
        if ( (unsigned __int64)v34 > 0xF )
        {
          nb = (size_t)v34;
          v36 = sub_22409D0(&v108, &v105, 0);
          v35 = nb;
          v108 = v36;
          v37 = (_QWORD *)v36;
          v110[0] = v105;
        }
        else
        {
          if ( v34 == (_BYTE *)1 )
          {
            LOBYTE(v110[0]) = *v32;
            v16 = (__int64)v110;
            goto LABEL_19;
          }
          if ( !v34 )
          {
            v16 = (__int64)v110;
            goto LABEL_19;
          }
          v37 = v110;
        }
        memcpy(v37, v32, v35);
        v34 = (_BYTE *)v105;
        v16 = v108;
LABEL_19:
        v109 = v34;
        v34[v16] = 0;
        v17 = sub_2241AC0(&v108, "grid_constant");
        if ( (_QWORD *)v108 != v110 )
        {
          na = v17;
          j_j___libc_free_0(v108, v110[0] + 1LL);
          v17 = na;
        }
        n = v33;
        if ( v17 )
        {
          v103[0] = v104;
          v18 = strlen(v32);
          v108 = v18;
          v19 = v18;
          if ( v18 > 0xF )
          {
            v103[0] = sub_22409D0(v103, &v108, 0);
            v53 = (_QWORD *)v103[0];
            v104[0] = v108;
          }
          else
          {
            if ( v18 == 1 )
            {
              LOBYTE(v104[0]) = *v32;
              v20 = v104;
              goto LABEL_25;
            }
            if ( !v18 )
            {
              v20 = v104;
              goto LABEL_25;
            }
            v53 = v104;
          }
          memcpy(v53, v32, v19);
          v18 = v108;
          v20 = (_QWORD *)v103[0];
LABEL_25:
          v103[1] = v18;
          *((_BYTE *)v20 + v18) = 0;
          if ( !(unsigned int)sub_2241AC0(v103, "preserve_n_data") )
          {
            if ( (_QWORD *)v103[0] != v104 )
              j_j___libc_free_0(v103[0], v104[0] + 1LL);
            v21 = a1[45];
LABEL_29:
            v22 = strlen(v32);
            v23 = sub_161FF10(v21, v32, v22);
            v24 = v101;
            v108 = v23;
            if ( v101 == v102 )
            {
              sub_1273E00((__int64)&v100, v101, &v108);
            }
            else
            {
              if ( v101 )
              {
                *(_QWORD *)v101 = v23;
                v24 = v101;
              }
              v101 = v24 + 8;
            }
            v25 = sub_1643350(a1[45]);
            v26 = sub_159C470(v25, n, 0);
            v29 = sub_1624210(v26, n, v27, v28);
            v30 = v101;
            v108 = v29;
            if ( v101 == v102 )
            {
              sub_1273E00((__int64)&v100, v101, &v108);
            }
            else
            {
              if ( v101 )
              {
                *(_QWORD *)v101 = v29;
                v30 = v101;
              }
              v101 = v30 + 8;
            }
            goto LABEL_37;
          }
          v105 = (__int64)v107;
          v54 = strlen(v32);
          v108 = v54;
          v55 = v54;
          if ( v54 > 0xF )
          {
            v105 = sub_22409D0(&v105, &v108, 0);
            v78 = (_QWORD *)v105;
            v107[0] = v108;
          }
          else
          {
            if ( v54 == 1 )
            {
              LOBYTE(v107[0]) = *v32;
              v56 = v107;
              goto LABEL_82;
            }
            if ( !v54 )
            {
              v56 = v107;
              goto LABEL_82;
            }
            v78 = v107;
          }
          memcpy(v78, v32, v55);
          v54 = v108;
          v56 = (_QWORD *)v105;
LABEL_82:
          v106 = v54;
          *((_BYTE *)v56 + v54) = 0;
          v86 = 1;
          if ( !(unsigned int)sub_2241AC0(&v105, "preserve_n_control") )
          {
LABEL_83:
            if ( (_QWORD *)v105 != v107 )
              j_j___libc_free_0(v105, v107[0] + 1LL);
            if ( (_QWORD *)v103[0] != v104 )
              j_j___libc_free_0(v103[0], v104[0] + 1LL);
            v21 = a1[45];
            if ( v86 )
              goto LABEL_29;
            v57 = strlen(v32);
            v58 = sub_161FF10(v21, v32, v57);
            v59 = v95;
            v108 = v58;
            if ( v95 == v96 )
            {
              sub_1273E00((__int64)&v94, v95, &v108);
            }
            else
            {
              if ( v95 )
              {
                *(_QWORD *)v95 = v58;
                v59 = v95;
              }
              v95 = v59 + 8;
            }
            LODWORD(v106) = 32;
            BYTE4(v106) = 0;
            v60 = a1[45];
            v105 = (unsigned int)n;
            v61 = sub_1643350(v60);
            v62 = sub_15A1070(v61, &v105);
            v65 = sub_1624210(v62, &v105, v63, v64);
            v108 = (__int64)v110;
            v93 = v65;
            v66 = (_BYTE *)strlen(v32);
            v103[0] = v66;
            v67 = (size_t)v66;
            if ( (unsigned __int64)v66 > 0xF )
            {
              v108 = sub_22409D0(&v108, v103, 0);
              v77 = (_QWORD *)v108;
              v110[0] = v103[0];
            }
            else
            {
              if ( v66 == (_BYTE *)1 )
              {
                LOBYTE(v110[0]) = *v32;
                v68 = (__int64)v110;
                goto LABEL_95;
              }
              if ( !v66 )
              {
                v68 = (__int64)v110;
                goto LABEL_95;
              }
              v77 = v110;
            }
            memcpy(v77, v32, v67);
            v66 = (_BYTE *)v103[0];
            v68 = v108;
LABEL_95:
            v109 = v66;
            v66[v68] = 0;
            v69 = sub_2241AC0(&v108, "full_custom_abi");
            if ( (_QWORD *)v108 != v110 )
              j_j___libc_free_0(v108, v110[0] + 1LL);
            if ( v69 )
            {
LABEL_98:
              v70 = v95;
              if ( v95 != v96 )
                goto LABEL_99;
LABEL_115:
              sub_126A620((__int64)&v94, v70, &v93);
              goto LABEL_102;
            }
            v108 = 0;
            v109 = 0;
            v71 = a1[45];
            v110[0] = 0;
            v72 = sub_161FF10(v71, "numParams", 9);
            v73 = v109;
            v74 = (_BYTE *)v110[0];
            v103[0] = v72;
            if ( v109 == (_BYTE *)v110[0] )
            {
              sub_1273E00((__int64)&v108, v109, v103);
              v75 = v109;
              if ( v109 == (_BYTE *)v110[0] )
                goto LABEL_136;
              if ( v109 )
                goto LABEL_111;
LABEL_112:
              v76 = v75 + 8;
              v109 = v76;
            }
            else
            {
              if ( v109 )
              {
                *(_QWORD *)v109 = v72;
                v73 = v109;
                v74 = (_BYTE *)v110[0];
              }
              v75 = v73 + 8;
              v109 = v75;
              if ( v75 != v74 )
              {
LABEL_111:
                *(_QWORD *)v75 = v93;
                v75 = v109;
                goto LABEL_112;
              }
LABEL_136:
              sub_126A620((__int64)&v108, v75, &v93);
              v76 = v109;
            }
            v93 = sub_1627350(a1[45], v108, (__int64)&v76[-v108] >> 3, 0, 1);
            if ( !v108 )
              goto LABEL_98;
            j_j___libc_free_0(v108, v110[0] - v108);
            v70 = v95;
            if ( v95 == v96 )
              goto LABEL_115;
LABEL_99:
            if ( v70 )
            {
              *(_QWORD *)v70 = v93;
              v70 = v95;
            }
            v95 = v70 + 8;
LABEL_102:
            if ( (unsigned int)v106 > 0x40 && v105 )
              j_j___libc_free_0_0(v105);
            goto LABEL_37;
          }
          v108 = (__int64)v110;
          v79 = (_BYTE *)strlen(v32);
          v93 = (__int64)v79;
          v80 = (size_t)v79;
          if ( (unsigned __int64)v79 > 0xF )
          {
            v108 = sub_22409D0(&v108, &v93, 0);
            v82 = (_QWORD *)v108;
            v110[0] = v93;
          }
          else
          {
            if ( v79 == (_BYTE *)1 )
            {
              LOBYTE(v110[0]) = *v32;
              v81 = (__int64)v110;
              goto LABEL_126;
            }
            if ( !v79 )
            {
              v81 = (__int64)v110;
              goto LABEL_126;
            }
            v82 = v110;
          }
          memcpy(v82, v32, v80);
          v79 = (_BYTE *)v93;
          v81 = v108;
LABEL_126:
          v109 = v79;
          v79[v81] = 0;
          v86 = (unsigned int)sub_2241AC0(&v108, "preserve_n_after") == 0;
          if ( (_QWORD *)v108 != v110 )
            j_j___libc_free_0(v108, v110[0] + 1LL);
          goto LABEL_83;
        }
        v38 = sub_1643350(a1[45]);
        v39 = sub_159C470(v38, v33, 0);
        v42 = sub_1624210(v39, v33, v40, v41);
        v43 = v98;
        v108 = v42;
        if ( v98 != v99 )
          break;
        sub_126A620((__int64)&v97, v98, &v108);
LABEL_37:
        v15 += 16;
        if ( v15 == v87 )
          goto LABEL_49;
      }
      if ( v98 )
      {
        *(_QWORD *)v98 = v42;
        v43 = v98;
      }
      v15 += 16;
      v98 = v43 + 8;
    }
    while ( v15 != v87 );
LABEL_49:
    if ( v98 != v97 )
    {
      v44 = sub_161FF10(a1[45], "grid_constant", 13);
      v45 = v95;
      v108 = v44;
      if ( v95 == v96 )
      {
        sub_1273E00((__int64)&v94, v95, &v108);
      }
      else
      {
        if ( v95 )
        {
          *(_QWORD *)v95 = v44;
          v45 = v95;
        }
        v95 = v45 + 8;
      }
      v46 = sub_1627350(a1[45], v97, (v98 - v97) >> 3, 0, 1);
      v47 = v95;
      v108 = v46;
      if ( v95 == v96 )
      {
        sub_1273E00((__int64)&v94, v95, &v108);
      }
      else
      {
        if ( v95 )
        {
          *(_QWORD *)v95 = v46;
          v47 = v95;
        }
        v95 = v47 + 8;
      }
    }
    if ( v101 == v100 )
      goto LABEL_129;
    v48 = sub_161FF10(a1[45], "preserve_reg_abi", 16);
    v49 = v95;
    v108 = v48;
    if ( v95 == v96 )
    {
      sub_1273E00((__int64)&v94, v95, &v108);
    }
    else
    {
      if ( v95 )
      {
        *(_QWORD *)v95 = v48;
        v49 = v95;
      }
      v95 = v49 + 8;
    }
    v50 = sub_1627350(a1[45], v100, (v101 - v100) >> 3, 0, 1);
    v51 = v95;
    v108 = v50;
    if ( v95 == v96 )
    {
      sub_1273E00((__int64)&v94, v95, &v108);
      goto LABEL_129;
    }
    if ( v95 )
    {
      *(_QWORD *)v95 = v50;
      v51 = v95;
    }
    v52 = v51 + 8;
    v95 = v52;
LABEL_67:
    v4 = sub_1627350(a1[45], v94, (__int64)&v52[-v94] >> 3, 0, 1);
    sub_1623CA0(v84, v4);
    if ( v100 )
    {
      v4 = v102 - v100;
      j_j___libc_free_0(v100, v102 - v100);
    }
    if ( v97 )
    {
      v4 = v99 - v97;
      j_j___libc_free_0(v97, v99 - v97);
    }
    if ( v94 )
    {
      v4 = (unsigned __int64)&v96[-v94];
      j_j___libc_free_0(v94, &v96[-v94]);
    }
    if ( v85 != v83 )
    {
      v85 += 8;
      v5 = a1[56];
      continue;
    }
    break;
  }
}
