// Function: sub_914410
// Address: 0x914410
//
void __fastcall sub_914410(__int64 *a1)
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
  __int64 v76; // rax
  _BYTE *v77; // rsi
  _QWORD *v78; // rdi
  _QWORD *v79; // rdi
  _BYTE *v80; // rax
  size_t v81; // r15
  __int64 v82; // rdx
  _QWORD *v83; // rdi
  __int64 v84; // [rsp+0h] [rbp-160h]
  __int64 v85; // [rsp+8h] [rbp-158h]
  __int64 v86; // [rsp+10h] [rbp-150h]
  bool v87; // [rsp+27h] [rbp-139h]
  __int64 v88; // [rsp+28h] [rbp-138h]
  _QWORD *v89; // [rsp+30h] [rbp-130h]
  int na; // [rsp+50h] [rbp-110h]
  size_t n; // [rsp+50h] [rbp-110h]
  size_t nb; // [rsp+50h] [rbp-110h]
  __int64 v94; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v95; // [rsp+70h] [rbp-F0h] BYREF
  _BYTE *v96; // [rsp+78h] [rbp-E8h]
  _BYTE *v97; // [rsp+80h] [rbp-E0h]
  _BYTE *v98; // [rsp+90h] [rbp-D0h] BYREF
  _BYTE *v99; // [rsp+98h] [rbp-C8h]
  _BYTE *v100; // [rsp+A0h] [rbp-C0h]
  _BYTE *v101; // [rsp+B0h] [rbp-B0h] BYREF
  _BYTE *v102; // [rsp+B8h] [rbp-A8h]
  _BYTE *v103; // [rsp+C0h] [rbp-A0h]
  _QWORD v104[2]; // [rsp+D0h] [rbp-90h] BYREF
  _QWORD v105[2]; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v106; // [rsp+F0h] [rbp-70h] BYREF
  size_t v107; // [rsp+F8h] [rbp-68h]
  _QWORD v108[2]; // [rsp+100h] [rbp-60h] BYREF
  __int64 v109; // [rsp+110h] [rbp-50h] BYREF
  _BYTE *v110; // [rsp+118h] [rbp-48h]
  _QWORD v111[8]; // [rsp+120h] [rbp-40h] BYREF

  if ( a1[54] == a1[55] )
    return;
  v1 = *a1;
  v2 = 0;
  v3 = off_4C5D0E8;
  if ( off_4C5D0E8 )
    v2 = strlen(off_4C5D0E8);
  v4 = (unsigned __int64)v3;
  v85 = sub_BA8E40(v1, v3, v2);
  v5 = a1[54];
  v6 = (a1[55] - v5) >> 3;
  if ( (int)v6 <= 0 )
    return;
  v7 = (unsigned int)(v6 - 1);
  v86 = 0;
  v84 = 8LL * (unsigned int)v7;
  while ( 2 )
  {
    v8 = *(_QWORD **)(v5 + v86);
    v97 = 0;
    v95 = 0;
    v9 = v8[4] - v8[3];
    v109 = 4;
    v10 = v8;
    v110 = 0;
    v11 = v8[2];
    v12 = v9 >> 4;
    v89 = v8;
    v111[0] = v11;
    v96 = 0;
    LOBYTE(v7) = v11 != -4096;
    if ( ((v11 != 0) & (unsigned __int8)v7) != 0 && v11 != -8192 )
    {
      v4 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      sub_BD6050(&v109, v4);
      v11 = v111[0];
    }
    v13 = sub_B98A20(v11, v4, v7, v10);
    v14 = v96;
    v106 = v13;
    if ( v96 == v97 )
    {
      sub_914280((__int64)&v95, v96, &v106);
    }
    else
    {
      if ( v96 )
      {
        *(_QWORD *)v96 = v13;
        v14 = v96;
      }
      v96 = v14 + 8;
    }
    if ( v111[0] != 0 && v111[0] != -4096 && v111[0] != -8192 )
      sub_BD60C0(&v109);
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v103 = 0;
    if ( (int)v12 <= 0 )
    {
LABEL_129:
      v52 = v96;
      goto LABEL_67;
    }
    v15 = 0;
    v88 = 16 * ((unsigned int)(v12 - 1) + 1LL);
    do
    {
      while ( 1 )
      {
        v31 = v15 + v89[3];
        v32 = *(const char **)v31;
        v33 = *(_DWORD *)(v31 + 8);
        v109 = (__int64)v111;
        if ( !v32 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v34 = (_BYTE *)strlen(v32);
        v106 = (__int64)v34;
        v35 = (size_t)v34;
        if ( (unsigned __int64)v34 > 0xF )
        {
          nb = (size_t)v34;
          v36 = sub_22409D0(&v109, &v106, 0);
          v35 = nb;
          v109 = v36;
          v37 = (_QWORD *)v36;
          v111[0] = v106;
        }
        else
        {
          if ( v34 == (_BYTE *)1 )
          {
            LOBYTE(v111[0]) = *v32;
            v16 = (__int64)v111;
            goto LABEL_19;
          }
          if ( !v34 )
          {
            v16 = (__int64)v111;
            goto LABEL_19;
          }
          v37 = v111;
        }
        memcpy(v37, v32, v35);
        v34 = (_BYTE *)v106;
        v16 = v109;
LABEL_19:
        v110 = v34;
        v34[v16] = 0;
        v17 = sub_2241AC0(&v109, "grid_constant");
        if ( (_QWORD *)v109 != v111 )
        {
          na = v17;
          j_j___libc_free_0(v109, v111[0] + 1LL);
          v17 = na;
        }
        n = v33;
        if ( v17 )
        {
          v104[0] = v105;
          v18 = strlen(v32);
          v109 = v18;
          v19 = v18;
          if ( v18 > 0xF )
          {
            v104[0] = sub_22409D0(v104, &v109, 0);
            v53 = (_QWORD *)v104[0];
            v105[0] = v109;
          }
          else
          {
            if ( v18 == 1 )
            {
              LOBYTE(v105[0]) = *v32;
              v20 = v105;
              goto LABEL_25;
            }
            if ( !v18 )
            {
              v20 = v105;
              goto LABEL_25;
            }
            v53 = v105;
          }
          memcpy(v53, v32, v19);
          v18 = v109;
          v20 = (_QWORD *)v104[0];
LABEL_25:
          v104[1] = v18;
          *((_BYTE *)v20 + v18) = 0;
          if ( !(unsigned int)sub_2241AC0(v104, "preserve_n_data") )
          {
            if ( (_QWORD *)v104[0] != v105 )
              j_j___libc_free_0(v104[0], v105[0] + 1LL);
            v21 = a1[43];
LABEL_29:
            v22 = strlen(v32);
            v23 = sub_B9B140(v21, v32, v22);
            v24 = v102;
            v109 = v23;
            if ( v102 == v103 )
            {
              sub_914280((__int64)&v101, v102, &v109);
            }
            else
            {
              if ( v102 )
              {
                *(_QWORD *)v102 = v23;
                v24 = v102;
              }
              v102 = v24 + 8;
            }
            v25 = sub_BCB2D0(a1[43]);
            v26 = sub_ACD640(v25, n, 0);
            v29 = sub_B98A20(v26, n, v27, v28);
            v30 = v102;
            v109 = v29;
            if ( v102 == v103 )
            {
              sub_914280((__int64)&v101, v102, &v109);
            }
            else
            {
              if ( v102 )
              {
                *(_QWORD *)v102 = v29;
                v30 = v102;
              }
              v102 = v30 + 8;
            }
            goto LABEL_37;
          }
          v106 = (__int64)v108;
          v54 = strlen(v32);
          v109 = v54;
          v55 = v54;
          if ( v54 > 0xF )
          {
            v106 = sub_22409D0(&v106, &v109, 0);
            v79 = (_QWORD *)v106;
            v108[0] = v109;
          }
          else
          {
            if ( v54 == 1 )
            {
              LOBYTE(v108[0]) = *v32;
              v56 = v108;
              goto LABEL_82;
            }
            if ( !v54 )
            {
              v56 = v108;
              goto LABEL_82;
            }
            v79 = v108;
          }
          memcpy(v79, v32, v55);
          v54 = v109;
          v56 = (_QWORD *)v106;
LABEL_82:
          v107 = v54;
          *((_BYTE *)v56 + v54) = 0;
          v87 = 1;
          if ( !(unsigned int)sub_2241AC0(&v106, "preserve_n_control") )
          {
LABEL_83:
            if ( (_QWORD *)v106 != v108 )
              j_j___libc_free_0(v106, v108[0] + 1LL);
            if ( (_QWORD *)v104[0] != v105 )
              j_j___libc_free_0(v104[0], v105[0] + 1LL);
            v21 = a1[43];
            if ( v87 )
              goto LABEL_29;
            v57 = strlen(v32);
            v58 = sub_B9B140(v21, v32, v57);
            v59 = v96;
            v109 = v58;
            if ( v96 == v97 )
            {
              sub_914280((__int64)&v95, v96, &v109);
            }
            else
            {
              if ( v96 )
              {
                *(_QWORD *)v96 = v58;
                v59 = v96;
              }
              v96 = v59 + 8;
            }
            LODWORD(v107) = 32;
            BYTE4(v107) = 0;
            v60 = a1[43];
            v106 = (unsigned int)n;
            v61 = sub_BCB2D0(v60);
            v62 = sub_AD8D80(v61, &v106);
            v65 = sub_B98A20(v62, &v106, v63, v64);
            v109 = (__int64)v111;
            v94 = v65;
            v66 = (_BYTE *)strlen(v32);
            v104[0] = v66;
            v67 = (size_t)v66;
            if ( (unsigned __int64)v66 > 0xF )
            {
              v109 = sub_22409D0(&v109, v104, 0);
              v78 = (_QWORD *)v109;
              v111[0] = v104[0];
            }
            else
            {
              if ( v66 == (_BYTE *)1 )
              {
                LOBYTE(v111[0]) = *v32;
                v68 = (__int64)v111;
                goto LABEL_95;
              }
              if ( !v66 )
              {
                v68 = (__int64)v111;
                goto LABEL_95;
              }
              v78 = v111;
            }
            memcpy(v78, v32, v67);
            v66 = (_BYTE *)v104[0];
            v68 = v109;
LABEL_95:
            v110 = v66;
            v66[v68] = 0;
            v69 = sub_2241AC0(&v109, "full_custom_abi");
            if ( (_QWORD *)v109 != v111 )
              j_j___libc_free_0(v109, v111[0] + 1LL);
            if ( v69 )
            {
LABEL_98:
              v70 = v96;
              if ( v96 != v97 )
                goto LABEL_99;
LABEL_115:
              sub_90AC40((__int64)&v95, v70, &v94);
              goto LABEL_102;
            }
            v109 = 0;
            v110 = 0;
            v71 = a1[43];
            v111[0] = 0;
            v72 = sub_B9B140(v71, "numParams", 9);
            v73 = v110;
            v74 = (_BYTE *)v111[0];
            v104[0] = v72;
            if ( v110 == (_BYTE *)v111[0] )
            {
              sub_914280((__int64)&v109, v110, v104);
              v75 = v110;
              if ( (_BYTE *)v111[0] == v110 )
                goto LABEL_136;
              if ( v110 )
              {
                v76 = v94;
                goto LABEL_111;
              }
LABEL_112:
              v77 = v75 + 8;
              v110 = v77;
            }
            else
            {
              if ( v110 )
              {
                *(_QWORD *)v110 = v72;
                v74 = (_BYTE *)v111[0];
                v73 = v110;
              }
              v75 = v73 + 8;
              v76 = v94;
              v110 = v75;
              if ( v74 != v75 )
              {
LABEL_111:
                *(_QWORD *)v75 = v76;
                v75 = v110;
                goto LABEL_112;
              }
LABEL_136:
              sub_90AC40((__int64)&v109, v75, &v94);
              v77 = v110;
            }
            v94 = sub_B9C770(a1[43], v109, (__int64)&v77[-v109] >> 3, 0, 1);
            if ( !v109 )
              goto LABEL_98;
            j_j___libc_free_0(v109, v111[0] - v109);
            v70 = v96;
            if ( v96 == v97 )
              goto LABEL_115;
LABEL_99:
            if ( v70 )
            {
              *(_QWORD *)v70 = v94;
              v70 = v96;
            }
            v96 = v70 + 8;
LABEL_102:
            if ( (unsigned int)v107 > 0x40 && v106 )
              j_j___libc_free_0_0(v106);
            goto LABEL_37;
          }
          v109 = (__int64)v111;
          v80 = (_BYTE *)strlen(v32);
          v94 = (__int64)v80;
          v81 = (size_t)v80;
          if ( (unsigned __int64)v80 > 0xF )
          {
            v109 = sub_22409D0(&v109, &v94, 0);
            v83 = (_QWORD *)v109;
            v111[0] = v94;
          }
          else
          {
            if ( v80 == (_BYTE *)1 )
            {
              LOBYTE(v111[0]) = *v32;
              v82 = (__int64)v111;
              goto LABEL_126;
            }
            if ( !v80 )
            {
              v82 = (__int64)v111;
              goto LABEL_126;
            }
            v83 = v111;
          }
          memcpy(v83, v32, v81);
          v80 = (_BYTE *)v94;
          v82 = v109;
LABEL_126:
          v110 = v80;
          v80[v82] = 0;
          v87 = (unsigned int)sub_2241AC0(&v109, "preserve_n_after") == 0;
          if ( (_QWORD *)v109 != v111 )
            j_j___libc_free_0(v109, v111[0] + 1LL);
          goto LABEL_83;
        }
        v38 = sub_BCB2D0(a1[43]);
        v39 = sub_ACD640(v38, v33, 0);
        v42 = sub_B98A20(v39, v33, v40, v41);
        v43 = v99;
        v109 = v42;
        if ( v99 != v100 )
          break;
        sub_90AC40((__int64)&v98, v99, &v109);
LABEL_37:
        v15 += 16;
        if ( v15 == v88 )
          goto LABEL_49;
      }
      if ( v99 )
      {
        *(_QWORD *)v99 = v42;
        v43 = v99;
      }
      v15 += 16;
      v99 = v43 + 8;
    }
    while ( v15 != v88 );
LABEL_49:
    if ( v99 != v98 )
    {
      v44 = sub_B9B140(a1[43], "grid_constant", 13);
      v45 = v96;
      v109 = v44;
      if ( v96 == v97 )
      {
        sub_914280((__int64)&v95, v96, &v109);
      }
      else
      {
        if ( v96 )
        {
          *(_QWORD *)v96 = v44;
          v45 = v96;
        }
        v96 = v45 + 8;
      }
      v46 = sub_B9C770(a1[43], v98, (v99 - v98) >> 3, 0, 1);
      v47 = v96;
      v109 = v46;
      if ( v96 == v97 )
      {
        sub_914280((__int64)&v95, v96, &v109);
      }
      else
      {
        if ( v96 )
        {
          *(_QWORD *)v96 = v46;
          v47 = v96;
        }
        v96 = v47 + 8;
      }
    }
    if ( v101 == v102 )
      goto LABEL_129;
    v48 = sub_B9B140(a1[43], "preserve_reg_abi", 16);
    v49 = v96;
    v109 = v48;
    if ( v96 == v97 )
    {
      sub_914280((__int64)&v95, v96, &v109);
    }
    else
    {
      if ( v96 )
      {
        *(_QWORD *)v96 = v48;
        v49 = v96;
      }
      v96 = v49 + 8;
    }
    v50 = sub_B9C770(a1[43], v101, (v102 - v101) >> 3, 0, 1);
    v51 = v96;
    v109 = v50;
    if ( v96 == v97 )
    {
      sub_914280((__int64)&v95, v96, &v109);
      goto LABEL_129;
    }
    if ( v96 )
    {
      *(_QWORD *)v96 = v50;
      v51 = v96;
    }
    v52 = v51 + 8;
    v96 = v52;
LABEL_67:
    v4 = sub_B9C770(a1[43], v95, (__int64)&v52[-v95] >> 3, 0, 1);
    sub_B979A0(v85, v4);
    if ( v101 )
    {
      v4 = v103 - v101;
      j_j___libc_free_0(v101, v103 - v101);
    }
    if ( v98 )
    {
      v4 = v100 - v98;
      j_j___libc_free_0(v98, v100 - v98);
    }
    if ( v95 )
    {
      v4 = (unsigned __int64)&v97[-v95];
      j_j___libc_free_0(v95, &v97[-v95]);
    }
    if ( v86 != v84 )
    {
      v86 += 8;
      v5 = a1[54];
      continue;
    }
    break;
  }
}
