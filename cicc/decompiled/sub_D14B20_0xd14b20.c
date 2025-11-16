// Function: sub_D14B20
// Address: 0xd14b20
//
__int64 __fastcall sub_D14B20(
        __int64 a1,
        int a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        unsigned __int8 a7)
{
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  unsigned int v23; // edx
  __int64 v24; // rax
  unsigned int v25; // esi
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // r14
  unsigned int v29; // edx
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // r14
  unsigned int v33; // eax
  __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  unsigned int v36; // eax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned int v39; // eax
  unsigned int v40; // eax
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // rax
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned int v54; // edx
  unsigned __int64 v55; // rax
  unsigned int v56; // esi
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  bool v63; // cc
  unsigned int v65; // [rsp+20h] [rbp-160h]
  unsigned int v66; // [rsp+20h] [rbp-160h]
  unsigned int v67; // [rsp+20h] [rbp-160h]
  const void **v68; // [rsp+28h] [rbp-158h]
  __int64 *v69; // [rsp+38h] [rbp-148h]
  __int64 v71; // [rsp+50h] [rbp-130h] BYREF
  unsigned int v72; // [rsp+58h] [rbp-128h]
  __int64 v73; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v74; // [rsp+68h] [rbp-118h]
  __int64 v75; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v76; // [rsp+78h] [rbp-108h]
  __int64 v77; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v78; // [rsp+88h] [rbp-F8h]
  __int64 v79; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v80; // [rsp+98h] [rbp-E8h]
  __int64 v81; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v82; // [rsp+A8h] [rbp-D8h]
  __int64 v83; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v84; // [rsp+B8h] [rbp-C8h]
  __int64 v85; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned int v86; // [rsp+C8h] [rbp-B8h]
  __int64 v87; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v88; // [rsp+D8h] [rbp-A8h]
  __int64 v89; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned int v90; // [rsp+E8h] [rbp-98h]
  __int64 v91; // [rsp+F0h] [rbp-90h] BYREF
  unsigned int v92; // [rsp+F8h] [rbp-88h]
  __int64 v93; // [rsp+100h] [rbp-80h] BYREF
  unsigned int v94; // [rsp+108h] [rbp-78h]
  __int64 v95; // [rsp+110h] [rbp-70h] BYREF
  unsigned int v96; // [rsp+118h] [rbp-68h]
  unsigned __int64 v97; // [rsp+120h] [rbp-60h] BYREF
  unsigned int v98; // [rsp+128h] [rbp-58h]
  __int64 v99; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v100; // [rsp+138h] [rbp-48h]
  __int64 v101; // [rsp+140h] [rbp-40h] BYREF
  unsigned int v102; // [rsp+148h] [rbp-38h]

  v69 = (__int64 *)(a5 + 16);
  v68 = (const void **)(a4 + 16);
  v9 = *(_DWORD *)(a4 + 24);
  v100 = v9;
  if ( v9 <= 0x40 )
  {
    v10 = *(_QWORD *)(a4 + 16);
LABEL_3:
    v11 = *(_QWORD *)(a5 + 16) & v10;
    v99 = v11;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v99, v68);
  v9 = v100;
  if ( v100 <= 0x40 )
  {
    v10 = v99;
    goto LABEL_3;
  }
  sub_C43B90(&v99, v69);
  v9 = v100;
  v11 = v99;
LABEL_4:
  v12 = *(_DWORD *)(a4 + 8);
  v102 = v9;
  v101 = v11;
  v100 = 0;
  v96 = v12;
  if ( v12 <= 0x40 )
  {
    v13 = *(_QWORD *)a4;
LABEL_6:
    v14 = *(_QWORD *)a5 & v13;
    v95 = v14;
    goto LABEL_7;
  }
  sub_C43780((__int64)&v95, (const void **)a4);
  v12 = v96;
  if ( v96 <= 0x40 )
  {
    v13 = v95;
    v9 = v102;
    goto LABEL_6;
  }
  sub_C43B90(&v95, (__int64 *)a5);
  v12 = v96;
  v14 = v95;
  v9 = v102;
LABEL_7:
  v98 = v12;
  v97 = v14;
  v96 = 0;
  if ( v9 > 0x40 )
  {
    sub_C43BD0(&v101, (__int64 *)&v97);
    v9 = v102;
    v15 = v101;
    v12 = v98;
  }
  else
  {
    v15 = v101 | v14;
    v101 = v15;
  }
  v72 = v9;
  v71 = v15;
  v102 = 0;
  if ( v12 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  sub_C48440((__int64)&v73, (unsigned __int8 *)&v71);
  sub_C48440((__int64)&v75, a3);
  v16 = v74;
  v98 = v74;
  if ( v74 <= 0x40 )
  {
    v17 = v73;
LABEL_23:
    v98 = 0;
    v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v17;
    if ( !v16 )
      v18 = 0;
    v97 = v18;
    goto LABEL_26;
  }
  sub_C43780((__int64)&v97, (const void **)&v73);
  v16 = v98;
  if ( v98 <= 0x40 )
  {
    v17 = v97;
    goto LABEL_23;
  }
  sub_C43D10((__int64)&v97);
  v16 = v98;
  v18 = v97;
  v98 = 0;
  v100 = v16;
  v99 = v97;
  if ( v16 > 0x40 )
  {
    sub_C43BD0(&v99, &v75);
    v16 = v100;
    v19 = v99;
    goto LABEL_27;
  }
LABEL_26:
  v19 = v75 | v18;
  v99 = v19;
LABEL_27:
  v102 = v16;
  v101 = v19;
  v100 = 0;
  sub_C45EE0((__int64)&v101, &v75);
  v78 = v102;
  v77 = v101;
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  v20 = v74;
  v100 = v74;
  if ( v74 <= 0x40 )
  {
    v21 = v73;
    goto LABEL_35;
  }
  sub_C43780((__int64)&v99, (const void **)&v73);
  v20 = v100;
  if ( v100 <= 0x40 )
  {
    v21 = v99;
LABEL_35:
    v22 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & ~v21;
    if ( !v20 )
      v22 = 0;
    goto LABEL_37;
  }
  sub_C43D10((__int64)&v99);
  v20 = v100;
  v22 = v99;
  v100 = 0;
  v102 = v20;
  v101 = v99;
  if ( v20 <= 0x40 )
  {
LABEL_37:
    v80 = v20;
    v79 = v77 ^ v22;
    goto LABEL_38;
  }
  sub_C43C10(&v101, &v77);
  v80 = v102;
  v79 = v101;
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
LABEL_38:
  sub_C48440((__int64)&v81, (unsigned __int8 *)&v79);
  v84 = 1;
  v83 = 0;
  v86 = 1;
  v85 = 0;
  if ( a2 )
  {
    v54 = *(_DWORD *)(a4 + 8);
    v100 = v54;
    if ( v54 > 0x40 )
    {
      sub_C43780((__int64)&v99, (const void **)a4);
      v54 = v100;
      if ( v100 > 0x40 )
      {
        sub_C43D10((__int64)&v99);
        v54 = v100;
        v57 = v99;
        v100 = 0;
        v102 = v54;
        v101 = v99;
        if ( v54 > 0x40 )
        {
          sub_C43BD0(&v101, (__int64 *)a5);
          v54 = v102;
          v102 = 0;
          v59 = v101;
          if ( v84 <= 0x40 )
            goto LABEL_192;
LABEL_163:
          if ( v83 )
          {
            v67 = v54;
            j_j___libc_free_0_0(v83);
            v83 = v59;
            v84 = v67;
            if ( v102 > 0x40 && v101 )
            {
              j_j___libc_free_0_0(v101);
              if ( v100 <= 0x40 )
                goto LABEL_167;
LABEL_194:
              if ( v99 )
              {
                j_j___libc_free_0_0(v99);
                v29 = *(_DWORD *)(a4 + 24);
                v100 = v29;
                if ( v29 <= 0x40 )
                  goto LABEL_168;
                goto LABEL_196;
              }
LABEL_167:
              v29 = *(_DWORD *)(a4 + 24);
              v100 = v29;
              if ( v29 <= 0x40 )
              {
LABEL_168:
                v60 = *(_QWORD *)(a4 + 16);
                goto LABEL_169;
              }
LABEL_196:
              sub_C43780((__int64)&v99, v68);
              v29 = v100;
              if ( v100 > 0x40 )
              {
                sub_C43D10((__int64)&v99);
                v29 = v100;
                v61 = v99;
                v100 = 0;
                v102 = v29;
                v101 = v99;
                if ( v29 > 0x40 )
                {
                  sub_C43BD0(&v101, v69);
                  v29 = v102;
                  v32 = v101;
                  goto LABEL_55;
                }
LABEL_172:
                v62 = *(_QWORD *)(a5 + 16) | v61;
                v102 = 0;
                v101 = v62;
                v32 = v62;
                if ( v86 <= 0x40 )
                  goto LABEL_173;
                goto LABEL_56;
              }
              v60 = v99;
LABEL_169:
              v100 = 0;
              v61 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v29) & ~v60;
              if ( !v29 )
                v61 = 0;
              v99 = v61;
              goto LABEL_172;
            }
LABEL_193:
            if ( v100 <= 0x40 )
              goto LABEL_167;
            goto LABEL_194;
          }
LABEL_192:
          v83 = v59;
          v84 = v54;
          goto LABEL_193;
        }
        v56 = v84;
LABEL_162:
        v58 = *(_QWORD *)a5 | v57;
        v102 = 0;
        v101 = v58;
        v59 = v58;
        if ( v56 <= 0x40 )
          goto LABEL_192;
        goto LABEL_163;
      }
      v55 = v99;
      v56 = v84;
    }
    else
    {
      v55 = *(_QWORD *)a4;
      v56 = 1;
    }
    v100 = 0;
    v57 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v54) & ~v55;
    if ( !v54 )
      v57 = 0;
    v99 = v57;
    goto LABEL_162;
  }
  v23 = *(_DWORD *)(a5 + 8);
  v100 = v23;
  if ( v23 <= 0x40 )
  {
    v24 = *(_QWORD *)a5;
    v25 = 1;
LABEL_41:
    v100 = 0;
    v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & ~v24;
    if ( !v23 )
      v26 = 0;
    v99 = v26;
    goto LABEL_44;
  }
  sub_C43780((__int64)&v99, (const void **)a5);
  v23 = v100;
  if ( v100 <= 0x40 )
  {
    v24 = v99;
    v25 = v84;
    goto LABEL_41;
  }
  sub_C43D10((__int64)&v99);
  v23 = v100;
  v26 = v99;
  v100 = 0;
  v102 = v23;
  v101 = v99;
  if ( v23 > 0x40 )
  {
    sub_C43BD0(&v101, (__int64 *)a4);
    v23 = v102;
    v102 = 0;
    v28 = v101;
    if ( v84 <= 0x40 )
      goto LABEL_182;
    goto LABEL_45;
  }
  v25 = v84;
LABEL_44:
  v27 = *(_QWORD *)a4 | v26;
  v102 = 0;
  v101 = v27;
  v28 = v27;
  if ( v25 <= 0x40 )
  {
LABEL_182:
    v83 = v28;
    v84 = v23;
    goto LABEL_183;
  }
LABEL_45:
  if ( !v83 )
    goto LABEL_182;
  v65 = v23;
  j_j___libc_free_0_0(v83);
  v83 = v28;
  v84 = v65;
  if ( v102 > 0x40 && v101 )
  {
    j_j___libc_free_0_0(v101);
    if ( v100 <= 0x40 )
      goto LABEL_49;
    goto LABEL_184;
  }
LABEL_183:
  if ( v100 <= 0x40 )
    goto LABEL_49;
LABEL_184:
  if ( !v99 )
  {
LABEL_49:
    v29 = *(_DWORD *)(a5 + 24);
    v100 = v29;
    if ( v29 <= 0x40 )
    {
LABEL_50:
      v30 = *(_QWORD *)(a5 + 16);
LABEL_51:
      v100 = 0;
      v31 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v29) & ~v30;
      if ( !v29 )
        v31 = 0;
      v99 = v31;
      goto LABEL_54;
    }
    goto LABEL_186;
  }
  j_j___libc_free_0_0(v99);
  v29 = *(_DWORD *)(a5 + 24);
  v100 = v29;
  if ( v29 <= 0x40 )
    goto LABEL_50;
LABEL_186:
  sub_C43780((__int64)&v99, (const void **)v69);
  v29 = v100;
  if ( v100 <= 0x40 )
  {
    v30 = v99;
    goto LABEL_51;
  }
  sub_C43D10((__int64)&v99);
  v29 = v100;
  v31 = v99;
  v100 = 0;
  v102 = v29;
  v101 = v99;
  if ( v29 <= 0x40 )
  {
LABEL_54:
    v101 = *(_QWORD *)(a4 + 16) | v31;
    v32 = v101;
    goto LABEL_55;
  }
  sub_C43BD0(&v101, (__int64 *)v68);
  v29 = v102;
  v32 = v101;
LABEL_55:
  v102 = 0;
  if ( v86 <= 0x40 )
  {
LABEL_173:
    v85 = v32;
    v86 = v29;
    goto LABEL_60;
  }
LABEL_56:
  if ( !v85 )
    goto LABEL_173;
  v66 = v29;
  j_j___libc_free_0_0(v85);
  v85 = v32;
  v86 = v66;
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
LABEL_60:
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  v33 = *(_DWORD *)(a5 + 8);
  v98 = v33;
  if ( v33 > 0x40 )
  {
    sub_C43780((__int64)&v97, (const void **)a5);
    v33 = v98;
    if ( v98 > 0x40 )
    {
      sub_C43D10((__int64)&v97);
      v33 = v98;
      v35 = v97;
      goto LABEL_68;
    }
    v34 = v97;
  }
  else
  {
    v34 = *(_QWORD *)a5;
  }
  v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v33) & ~v34;
  if ( !v33 )
    v35 = 0;
  v97 = v35;
LABEL_68:
  v100 = v33;
  v36 = *(_DWORD *)(a4 + 8);
  v99 = v35;
  v98 = 0;
  v94 = v36;
  if ( v36 > 0x40 )
  {
    sub_C43780((__int64)&v93, (const void **)a4);
    v36 = v94;
    if ( v94 > 0x40 )
    {
      sub_C43D10((__int64)&v93);
      v36 = v94;
      v38 = v93;
      goto LABEL_73;
    }
    v37 = v93;
  }
  else
  {
    v37 = *(_QWORD *)a4;
  }
  v38 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v36) & ~v37;
  if ( !v36 )
    v38 = 0;
  v93 = v38;
LABEL_73:
  v95 = v38;
  v96 = v36;
  v94 = 0;
  sub_C45EE0((__int64)&v99, &v95);
  v39 = v100;
  v100 = 0;
  v102 = v39;
  v101 = v99;
  sub_C46A40((__int64)&v101, a6 ^ 1u);
  v88 = v102;
  v87 = v101;
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  v100 = *(_DWORD *)(a4 + 24);
  if ( v100 > 0x40 )
    sub_C43780((__int64)&v99, v68);
  else
    v99 = *(_QWORD *)(a4 + 16);
  sub_C45EE0((__int64)&v99, v69);
  v40 = v100;
  v100 = 0;
  v102 = v40;
  v101 = v99;
  sub_C46A40((__int64)&v101, a7);
  v41 = v102;
  v90 = v102;
  v89 = v101;
  if ( v100 > 0x40 && v99 )
  {
    j_j___libc_free_0_0(v99);
    v41 = v90;
  }
  v100 = v41;
  if ( v41 <= 0x40 )
  {
    v42 = v89;
LABEL_92:
    v43 = v85 | v42;
    v99 = v43;
    goto LABEL_93;
  }
  sub_C43780((__int64)&v99, (const void **)&v89);
  v41 = v100;
  if ( v100 <= 0x40 )
  {
    v42 = v99;
    goto LABEL_92;
  }
  sub_C43BD0(&v99, &v85);
  v41 = v100;
  v43 = v99;
LABEL_93:
  v44 = v88;
  v102 = v41;
  v101 = v43;
  v100 = 0;
  v94 = v88;
  if ( v88 <= 0x40 )
  {
    v45 = v87;
LABEL_95:
    v94 = 0;
    v46 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v44) & ~v45;
    if ( !v44 )
      v46 = 0;
    v93 = v46;
    goto LABEL_98;
  }
  sub_C43780((__int64)&v93, (const void **)&v87);
  v44 = v94;
  if ( v94 <= 0x40 )
  {
    v45 = v93;
    v41 = v102;
    goto LABEL_95;
  }
  sub_C43D10((__int64)&v93);
  v44 = v94;
  v46 = v93;
  v94 = 0;
  v96 = v44;
  v95 = v93;
  if ( v44 <= 0x40 )
  {
    v41 = v102;
LABEL_98:
    v47 = v83 | v46;
    v95 = v47;
    goto LABEL_99;
  }
  sub_C43BD0(&v95, &v83);
  v44 = v96;
  v47 = v95;
  v41 = v102;
LABEL_99:
  v98 = v44;
  v97 = v47;
  v96 = 0;
  if ( v41 > 0x40 )
  {
    sub_C43B90(&v101, (__int64 *)&v97);
    v41 = v102;
    v48 = v101;
    v44 = v98;
  }
  else
  {
    v48 = v101 & v47;
    v101 = v48;
  }
  v92 = v41;
  v91 = v48;
  v102 = 0;
  if ( v44 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  v49 = v82;
  v100 = v82;
  if ( v82 <= 0x40 )
  {
    v50 = v81;
LABEL_118:
    v51 = v91 & v50;
LABEL_119:
    v52 = *(_QWORD *)a3 | v51;
    *(_DWORD *)(a1 + 8) = v49;
    *(_QWORD *)a1 = v52;
    goto LABEL_120;
  }
  sub_C43780((__int64)&v99, (const void **)&v81);
  v49 = v100;
  if ( v100 <= 0x40 )
  {
    v50 = v99;
    goto LABEL_118;
  }
  sub_C43B90(&v99, &v91);
  v49 = v100;
  v51 = v99;
  v100 = 0;
  v102 = v49;
  v101 = v99;
  if ( v49 <= 0x40 )
    goto LABEL_119;
  sub_C43BD0(&v101, (__int64 *)a3);
  v63 = v100 <= 0x40;
  *(_DWORD *)(a1 + 8) = v102;
  *(_QWORD *)a1 = v101;
  if ( !v63 && v99 )
    j_j___libc_free_0_0(v99);
LABEL_120:
  if ( v92 > 0x40 && v91 )
    j_j___libc_free_0_0(v91);
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
  if ( v88 > 0x40 && v87 )
    j_j___libc_free_0_0(v87);
  if ( v86 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
  if ( v84 > 0x40 && v83 )
    j_j___libc_free_0_0(v83);
  if ( v82 > 0x40 && v81 )
    j_j___libc_free_0_0(v81);
  if ( v80 > 0x40 && v79 )
    j_j___libc_free_0_0(v79);
  if ( v78 > 0x40 && v77 )
    j_j___libc_free_0_0(v77);
  if ( v76 > 0x40 && v75 )
    j_j___libc_free_0_0(v75);
  if ( v74 > 0x40 && v73 )
    j_j___libc_free_0_0(v73);
  if ( v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  return a1;
}
