// Function: sub_11EAE00
// Address: 0x11eae00
//
__int64 __fastcall sub_11EAE00(__int64 a1, __int64 a2, unsigned int **a3)
{
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // r12
  unsigned __int8 v6; // r13
  __int64 *v7; // r14
  _DWORD *v8; // r13
  char v9; // r8
  void *v10; // r13
  unsigned __int8 v11; // al
  __int64 v12; // r14
  unsigned __int8 *v14; // rax
  unsigned __int8 *v15; // rcx
  unsigned __int8 *v16; // rdx
  __int64 *v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int8 *v19; // rax
  bool v20; // r13
  __int64 v21; // rax
  _BYTE *v22; // rax
  __int64 *v23; // rax
  unsigned __int8 v24; // dl
  char v25; // al
  double v26; // xmm0_8
  double v27; // xmm0_8
  unsigned __int8 *v28; // rsi
  __int64 v29; // r13
  __int64 v30; // rdx
  char **v31; // rdi
  char **v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 *v35; // r13
  char *v36; // rax
  size_t v37; // rdx
  void **v38; // rdi
  bool v39; // al
  int v40; // eax
  char v41; // r8
  int v42; // ecx
  unsigned __int64 v43; // rax
  int v44; // eax
  double v45; // xmm0_8
  unsigned __int8 *v46; // rax
  __int64 v47; // r13
  char **v48; // rcx
  char *v49; // r14
  char **v50; // rbx
  char *v51; // rsi
  char *v52; // r12
  unsigned __int64 v53; // rax
  unsigned __int8 *v54; // r14
  int v55; // eax
  float v56; // xmm0_4
  float v57; // xmm0_4
  unsigned int v58; // ebx
  unsigned int v59; // r13d
  unsigned int v60; // r14d
  char *v61; // rax
  int v62; // r10d
  __int64 v63; // rdx
  char *v64; // r11
  _BYTE *v65; // rsi
  __int64 v66; // r15
  char *v67; // rax
  __int64 v68; // rdx
  __int64 v69; // [rsp+10h] [rbp-100h]
  bool v70; // [rsp+18h] [rbp-F8h]
  __int64 *v71; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v72; // [rsp+20h] [rbp-F0h]
  int v73; // [rsp+20h] [rbp-F0h]
  char v74; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v75; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v76; // [rsp+28h] [rbp-E8h]
  char v77; // [rsp+28h] [rbp-E8h]
  int v78; // [rsp+28h] [rbp-E8h]
  char v79; // [rsp+28h] [rbp-E8h]
  char *v80; // [rsp+28h] [rbp-E8h]
  char *v82; // [rsp+38h] [rbp-D8h]
  char *v83; // [rsp+40h] [rbp-D0h]
  bool v84; // [rsp+40h] [rbp-D0h]
  char v85; // [rsp+40h] [rbp-D0h]
  unsigned int v86; // [rsp+40h] [rbp-D0h]
  __int64 *v87; // [rsp+48h] [rbp-C8h]
  __int64 v88; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v90; // [rsp+50h] [rbp-C0h]
  __int64 v91; // [rsp+58h] [rbp-B8h]
  __int64 v92; // [rsp+58h] [rbp-B8h]
  bool v93; // [rsp+6Fh] [rbp-A1h] BYREF
  __int64 v94; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v95; // [rsp+78h] [rbp-98h] BYREF
  unsigned __int64 v96; // [rsp+80h] [rbp-90h] BYREF
  __int64 v97; // [rsp+88h] [rbp-88h]
  unsigned __int8 *v98; // [rsp+90h] [rbp-80h] BYREF
  char **v99; // [rsp+98h] [rbp-78h]
  char *v100; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v101; // [rsp+B8h] [rbp-58h]
  __int16 v102; // [rsp+D0h] [rbp-40h]

  v3 = a2;
  v87 = (__int64 *)sub_B43CA0(a2);
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = *(_BYTE *)v5;
  v82 = *(char **)(a2 + 32 * (1 - v4));
  v91 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)v5 != 85 )
  {
LABEL_2:
    v7 = (__int64 *)(v5 + 24);
    if ( v6 == 18 )
      goto LABEL_3;
    goto LABEL_46;
  }
  v33 = *(_QWORD *)(v5 + 16);
  if ( !v33 )
    return 0;
  v12 = *(_QWORD *)(v33 + 8);
  if ( v12 || !sub_B45190(v5) || !sub_B45190(a2) )
    return 0;
  v34 = *(_QWORD *)(v5 - 32);
  if ( v34 && !*(_BYTE *)v34 && *(_QWORD *)(v34 + 24) == *(_QWORD *)(v5 + 80) )
  {
    v35 = *(__int64 **)(a1 + 24);
    v36 = (char *)sub_BD5D20(v34);
    if ( !(unsigned __int8)sub_980AF0(*v35, v36, v37, &v96) || !sub_11C99B0(v87, *(__int64 **)(a1 + 24), v96) )
    {
      v6 = *(_BYTE *)v5;
      goto LABEL_2;
    }
    if ( (unsigned int)v96 > 0xE9 )
    {
      if ( (unsigned int)(v96 - 234) > 1 )
        return v12;
    }
    else
    {
      if ( (unsigned int)v96 > 0xE6 )
      {
        v58 = 233;
        v59 = 232;
        v60 = 90;
        v67 = sub_11DD430(*(__int64 **)(a1 + 24), 0xE7u);
        v62 = 231;
        v92 = v68;
        v64 = v67;
        goto LABEL_144;
      }
      if ( (_DWORD)v96 != 227 )
        return v12;
    }
    v58 = 235;
    v59 = 234;
    v60 = 88;
    v61 = sub_11DD430(*(__int64 **)(a1 + 24), 0xE3u);
    v62 = 227;
    v92 = v63;
    v64 = v61;
LABEL_144:
    v100 = "mul";
    v102 = 259;
    v86 = v62;
    v80 = v64;
    v65 = *(_BYTE **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
    HIDWORD(v98) = 0;
    v66 = sub_A826E0(a3, v65, v82, (unsigned int)v98, (__int64)&v100, 0);
    if ( sub_B49E00(v5) )
    {
      BYTE4(v98) = 0;
      v102 = 261;
      v100 = v80;
      v101 = v92;
      v12 = sub_B33BC0((__int64)a3, v60, v66, (__int64)v98, (__int64)&v100);
    }
    else
    {
      v100 = *(char **)(v5 + 72);
      v12 = sub_11CCA60(v66, *(__int64 **)(a1 + 24), v86, v59, v58, (__int64)a3, (__int64 *)&v100);
    }
    sub_11EA700(a1);
    sub_11EADF0(a1);
    return v12;
  }
LABEL_46:
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17 > 1 )
    return 0;
  if ( v6 > 0x15u )
    return 0;
  v22 = sub_AD7630(v5, 0, v4);
  if ( !v22 || *v22 != 18 )
    return 0;
  v7 = (__int64 *)(v22 + 24);
LABEL_3:
  v94 = 0;
  v70 = sub_B49E00(a2);
  v8 = sub_C33320();
  if ( !v70 && (unsigned int)*(unsigned __int8 *)(v91 + 8) - 17 <= 1 )
    goto LABEL_5;
  sub_C3B1B0((__int64)&v100, 2.0);
  sub_C407B0(&v98, (__int64 *)&v100, v8);
  sub_C338F0((__int64)&v100);
  sub_C41640((__int64 *)&v98, (_DWORD *)*v7, 1, (bool *)&v100);
  v72 = v98;
  v75 = (unsigned __int8 *)*v7;
  v14 = (unsigned __int8 *)sub_C33340();
  v84 = 0;
  v15 = v72;
  v16 = v14;
  if ( v75 == v72 )
  {
    v76 = v14;
    if ( v14 != v72 )
    {
      v84 = sub_C33D00((__int64)v7, (__int64)&v98);
      if ( v76 != v98 )
        goto LABEL_30;
      goto LABEL_69;
    }
    v39 = sub_C3E590((__int64)v7, (__int64)&v98);
    v15 = v98;
    v16 = v76;
    v84 = v39;
  }
  if ( v16 != v15 )
  {
LABEL_30:
    sub_C338F0((__int64)&v98);
    goto LABEL_31;
  }
LABEL_69:
  if ( v99 )
  {
    v30 = (__int64)*(v99 - 1);
    v31 = &v99[3 * v30];
    if ( v99 != v31 )
    {
      v32 = &v99[3 * v30];
      do
      {
        v32 -= 3;
        sub_91D830(v32);
      }
      while ( v99 != v32 );
      v31 = v32;
      v3 = a2;
    }
    j_j_j___libc_free_0_0(v31 - 1);
  }
LABEL_31:
  if ( v84 && (unsigned __int8)(*v82 - 72) <= 1u )
  {
    v17 = *(__int64 **)(a1 + 24);
    if ( v70 )
    {
      v53 = sub_11DBA30(v82, (__int64)a3, *(_DWORD *)(*v17 + 172));
      if ( v53 )
      {
        v90 = v53;
        v54 = sub_AD8DD0(v91, 1.0);
        v102 = 259;
        v100 = "exp2";
        v55 = sub_B45210(v3);
        v98 = v54;
        LODWORD(v95) = v55;
        BYTE4(v95) = 1;
        v99 = (char **)v90;
        v96 = v91;
        v97 = *(_QWORD *)(v90 + 8);
        v12 = sub_B33D10((__int64)a3, 0xD1u, (__int64)&v96, 2, (int)&v98, 2, v95, (__int64)&v100);
        if ( !v12 )
          return v12;
        goto LABEL_60;
      }
    }
    else if ( sub_11C9D70(v87, v17, v91, 0x149u, 0x14Au, 0x14Bu) )
    {
      v18 = sub_11DBA30(v82, (__int64)a3, *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
      if ( v18 )
      {
        v88 = v18;
        v19 = sub_AD8DD0(v91, 1.0);
        v12 = sub_11CD140((__int64)v19, v88, *(__int64 **)(a1 + 24), 0x149u, 0x14Au, 0x14Bu, (__int64)a3, &v94);
        if ( v12 )
          goto LABEL_60;
        return 0;
      }
    }
  }
LABEL_5:
  if ( !sub_11C9D70(v87, *(__int64 **)(a1 + 24), v91, 0xE7u, 0xE8u, 0xE9u) )
    goto LABEL_20;
  sub_C3B1B0((__int64)&v100, 1.0);
  sub_C407B0(&v98, (__int64 *)&v100, v8);
  sub_C338F0((__int64)&v100);
  sub_C41640((__int64 *)&v98, (_DWORD *)*v7, 0, &v93);
  v83 = (char *)sub_C33340();
  if ( v98 == (unsigned __int8 *)v83 )
    sub_C3C790(&v100, (_QWORD **)&v98);
  else
    sub_C33EB0(&v100, (__int64 *)&v98);
  if ( v100 == v83 )
    sub_C3EF50(&v100, (__int64)v7, 1u);
  else
    sub_C3B6C0((__int64)&v100, (__int64)v7, 1);
  if ( v98 == (unsigned __int8 *)v83 )
  {
    if ( v83 == v100 )
    {
      if ( v99 )
      {
        v48 = &v99[3 * (_QWORD)*(v99 - 1)];
        if ( v99 != v48 )
        {
          v71 = v7;
          v49 = v100;
          v69 = v3;
          v50 = &v99[3 * (_QWORD)*(v99 - 1)];
          do
          {
            v50 -= 3;
            if ( *v50 == v49 )
            {
              v51 = v50[1];
              if ( v51 )
              {
                v52 = &v51[24 * *((_QWORD *)v51 - 1)];
                if ( v51 != v52 )
                {
                  do
                  {
                    v52 -= 24;
                    sub_91D830(v52);
                  }
                  while ( v50[1] != v52 );
                }
                j_j_j___libc_free_0_0(v52 - 8);
              }
            }
            else
            {
              sub_C338F0((__int64)v50);
            }
          }
          while ( v99 != v50 );
          v48 = v50;
          v7 = v71;
          v3 = v69;
        }
        j_j_j___libc_free_0_0(v48 - 1);
      }
      goto LABEL_134;
    }
  }
  else if ( v83 != v100 )
  {
    sub_C33870((__int64)&v98, (__int64)&v100);
    goto LABEL_13;
  }
  sub_91D830(&v98);
  if ( v83 != v100 )
  {
    sub_C338E0((__int64)&v98, (__int64)&v100);
    goto LABEL_13;
  }
LABEL_134:
  sub_C3C840(&v98, &v100);
LABEL_13:
  sub_91D830(&v100);
  if ( (char *)*v7 == v83 )
    v74 = sub_C405F0((__int64)v7);
  else
    v74 = sub_C3BCA0((__int64)v7);
  if ( v98 == (unsigned __int8 *)v83 )
    v9 = sub_C405F0((__int64)&v98);
  else
    v9 = sub_C3BCA0((__int64)&v98);
  LODWORD(v97) = 64;
  v96 = 0;
  BYTE4(v97) = 0;
  if ( v9 )
  {
    v38 = (void **)&v98;
  }
  else
  {
    if ( !v74 )
    {
LABEL_19:
      sub_91D830(&v98);
LABEL_20:
      sub_C3B1B0((__int64)&v100, 10.0);
      sub_C407B0(&v98, (__int64 *)&v100, v8);
      sub_C338F0((__int64)&v100);
      sub_C41640((__int64 *)&v98, (_DWORD *)*v7, 1, (bool *)&v100);
      if ( (unsigned __int8 *)*v7 == v98 )
      {
        v10 = (void *)*v7;
        if ( v10 == sub_C33340() )
          v20 = sub_C3E590((__int64)v7, (__int64)&v98);
        else
          v20 = sub_C33D00((__int64)v7, (__int64)&v98);
        sub_91D830(&v98);
        if ( v20 && sub_11C9D70(v87, *(__int64 **)(a1 + 24), v91, 0xE4u, 0xE5u, 0xE6u) )
        {
          if ( sub_B49E00(v3) )
          {
            v100 = "exp10";
            v102 = 259;
            LODWORD(v98) = sub_B45210(v3);
            BYTE4(v98) = 1;
            v96 = (unsigned __int64)v82;
            v95 = v91;
            v12 = sub_B33D10((__int64)a3, 0x59u, (__int64)&v95, 1, (int)&v96, 1, (__int64)v98, (__int64)&v100);
            if ( !v12 )
              return 0;
            goto LABEL_60;
          }
          v21 = sub_11CCA60((__int64)v82, *(__int64 **)(a1 + 24), 0xE4u, 0xE5u, 0xE6u, (__int64)a3, &v94);
LABEL_44:
          v12 = v21;
          if ( !v21 )
            return 0;
LABEL_60:
          if ( *(_BYTE *)v12 == 85 )
            *(_WORD *)(v12 + 2) = *(_WORD *)(v12 + 2) & 0xFFFC | *(_WORD *)(v3 + 2) & 3;
          return v12;
        }
      }
      else
      {
        sub_91D830(&v98);
      }
      if ( !(unsigned __int8)sub_B45200(v3) || !sub_B451C0(v3) )
        return 0;
      if ( (void *)*v7 == sub_C33340() )
      {
        v23 = (__int64 *)v7[1];
        v24 = *((_BYTE *)v23 + 20) & 7;
        if ( v24 <= 1u || v24 == 3 )
          return 0;
      }
      else
      {
        v11 = *((_BYTE *)v7 + 20) & 7;
        if ( v11 <= 1u || v11 == 3 )
          return 0;
        v23 = v7;
      }
      if ( (*((_BYTE *)v23 + 20) & 8) == 0 )
      {
        v25 = *(_BYTE *)(v91 + 8);
        if ( v25 == 2 )
        {
          v56 = sub_C41C30(v7, (__m128i)0x4024000000000000uLL);
          v57 = log2f(v56);
          v28 = sub_AD8DD0(v91, v57);
        }
        else
        {
          if ( v25 != 3 )
            return 0;
          v26 = sub_C41B00(v7);
          v27 = log2(v26);
          v28 = sub_AD8DD0(v91, v27);
        }
        if ( v28 )
        {
          HIDWORD(v98) = 0;
          v100 = "mul";
          v102 = 259;
          v29 = sub_A826E0(a3, v28, v82, (unsigned int)v98, (__int64)&v100, 0);
          if ( sub_B49E00(v3) )
          {
            BYTE4(v98) = 0;
            v100 = "exp2";
            v102 = 259;
            v12 = sub_B33BC0((__int64)a3, 0x5Au, v29, (__int64)v98, (__int64)&v100);
            if ( !v12 )
              return v12;
            goto LABEL_60;
          }
          if ( sub_11C9D70(v87, *(__int64 **)(a1 + 24), v91, 0xE7u, 0xE8u, 0xE9u) )
          {
            v21 = sub_11CCA60(v29, *(__int64 **)(a1 + 24), 0xE7u, 0xE8u, 0xE9u, (__int64)a3, &v94);
            goto LABEL_44;
          }
        }
      }
      return 0;
    }
    v38 = (void **)v7;
  }
  v77 = v9;
  if ( (unsigned int)sub_C41980(v38, (__int64)&v96, 0, &v93) )
    goto LABEL_86;
  BYTE4(v101) = 0;
  LODWORD(v101) = 64;
  v100 = (char *)1;
  v40 = sub_AA8A40((__int64 *)&v96, (__int64 *)&v100);
  v41 = v77;
  if ( (unsigned int)v101 > 0x40 && v100 )
  {
    v78 = v40;
    v85 = v41;
    j_j___libc_free_0_0(v100);
    v40 = v78;
    v41 = v85;
  }
  if ( v40 <= 0 )
  {
LABEL_86:
    if ( (unsigned int)v97 <= 0x40 )
      goto LABEL_19;
    goto LABEL_87;
  }
  v42 = v97;
  if ( (unsigned int)v97 > 0x40 )
  {
    v73 = v97;
    v79 = v41;
    if ( (unsigned int)sub_C44630((__int64)&v96) == 1 )
    {
      v44 = sub_C444A0((__int64)&v96);
      v42 = v73;
      v41 = v79;
      goto LABEL_104;
    }
LABEL_87:
    if ( v96 )
      j_j___libc_free_0_0(v96);
    goto LABEL_19;
  }
  if ( !v96 || (v96 & (v96 - 1)) != 0 )
    goto LABEL_19;
  _BitScanReverse64(&v43, v96);
  v44 = v97 + (v43 ^ 0x3F) - 64;
LABEL_104:
  v45 = (double)(v42 - 1 - v44);
  if ( v41 )
    v45 = -v45;
  v100 = "mul";
  v102 = 259;
  v46 = sub_AD8DD0(v91, v45);
  HIDWORD(v95) = 0;
  v47 = sub_A826E0(a3, v82, v46, (unsigned int)v95, (__int64)&v100, 0);
  if ( sub_B49E00(v3) )
  {
    BYTE4(v95) = 0;
    v100 = "exp2";
    v102 = 259;
    v12 = sub_B33BC0((__int64)a3, 0x5Au, v47, v95, (__int64)&v100);
    if ( v12 )
      goto LABEL_108;
  }
  else
  {
    v12 = sub_11CCA60(v47, *(__int64 **)(a1 + 24), 0xE7u, 0xE8u, 0xE9u, (__int64)a3, &v94);
    if ( v12 )
    {
LABEL_108:
      if ( *(_BYTE *)v12 == 85 )
        *(_WORD *)(v12 + 2) = *(_WORD *)(v12 + 2) & 0xFFFC | *(_WORD *)(v3 + 2) & 3;
    }
  }
  if ( (unsigned int)v97 > 0x40 && v96 )
    j_j___libc_free_0_0(v96);
  sub_91D830(&v98);
  return v12;
}
