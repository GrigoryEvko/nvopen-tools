// Function: sub_DD1050
// Address: 0xdd1050
//
_QWORD *__fastcall sub_DD1050(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v6; // r8
  __int64 v7; // rax
  __int64 v8; // r9
  signed __int64 v9; // r11
  signed __int64 v10; // r10
  __int64 v11; // rbx
  __int64 v12; // rdi
  unsigned int v13; // r15d
  int v14; // eax
  __int64 v15; // r10
  char *v16; // rcx
  __int64 *v17; // rdx
  __int64 v19; // rax
  char v20; // bl
  __int64 v21; // rax
  __int64 v22; // r12
  _QWORD *v23; // rax
  __int64 *v24; // rsi
  _QWORD *v25; // r12
  __int64 v26; // rax
  unsigned int *v27; // rdi
  __int64 v28; // rax
  unsigned int v29; // edx
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 *v32; // rax
  __int64 v33; // r15
  _QWORD *v34; // rax
  _QWORD *v35; // rax
  int v36; // eax
  unsigned int v37; // edx
  unsigned int v38; // ecx
  __int64 *v39; // rsi
  unsigned int v40; // r15d
  size_t n; // [rsp+0h] [rbp-210h]
  size_t na; // [rsp+0h] [rbp-210h]
  void *srca; // [rsp+8h] [rbp-208h]
  __int64 *srcb; // [rsp+8h] [rbp-208h]
  __int64 *src; // [rsp+8h] [rbp-208h]
  __int64 *v46; // [rsp+10h] [rbp-200h]
  __int64 v47; // [rsp+10h] [rbp-200h]
  int v48; // [rsp+10h] [rbp-200h]
  unsigned int v49; // [rsp+10h] [rbp-200h]
  __int64 v50; // [rsp+18h] [rbp-1F8h]
  __int64 v51; // [rsp+18h] [rbp-1F8h]
  int v52; // [rsp+18h] [rbp-1F8h]
  __int64 v53; // [rsp+18h] [rbp-1F8h]
  unsigned int v54; // [rsp+18h] [rbp-1F8h]
  unsigned int v55; // [rsp+24h] [rbp-1ECh] BYREF
  __int64 v56; // [rsp+28h] [rbp-1E8h] BYREF
  __int64 v57; // [rsp+30h] [rbp-1E0h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-1D8h]
  __int64 v59; // [rsp+40h] [rbp-1D0h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-1C8h]
  __int64 v61; // [rsp+50h] [rbp-1C0h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-1B8h]
  __int64 v63; // [rsp+60h] [rbp-1B0h] BYREF
  unsigned int v64; // [rsp+68h] [rbp-1A8h]
  const void *v65; // [rsp+70h] [rbp-1A0h] BYREF
  unsigned int v66; // [rsp+78h] [rbp-198h]
  void *v67; // [rsp+80h] [rbp-190h] BYREF
  unsigned int v68; // [rsp+88h] [rbp-188h]
  __int64 v69[2]; // [rsp+90h] [rbp-180h] BYREF
  char v70; // [rsp+A0h] [rbp-170h]
  __int64 v71[2]; // [rsp+B0h] [rbp-160h] BYREF
  char v72; // [rsp+C0h] [rbp-150h]
  __int64 v73[2]; // [rsp+D0h] [rbp-140h] BYREF
  char v74; // [rsp+E0h] [rbp-130h]
  void *v75; // [rsp+F0h] [rbp-120h] BYREF
  unsigned int v76; // [rsp+F8h] [rbp-118h]
  char v77; // [rsp+100h] [rbp-110h]
  __int64 v78; // [rsp+110h] [rbp-100h] BYREF
  unsigned int v79; // [rsp+118h] [rbp-F8h]
  char v80; // [rsp+120h] [rbp-F0h]
  char v81; // [rsp+128h] [rbp-E8h]
  const void *v82; // [rsp+130h] [rbp-E0h] BYREF
  unsigned int v83; // [rsp+138h] [rbp-D8h]
  char v84; // [rsp+140h] [rbp-D0h]
  char v85; // [rsp+148h] [rbp-C8h]
  unsigned int *v86; // [rsp+150h] [rbp-C0h] BYREF
  unsigned int *v87; // [rsp+158h] [rbp-B8h]
  __int64 *v88; // [rsp+160h] [rbp-B0h]
  __int64 *v89; // [rsp+168h] [rbp-A8h]
  __int64 *v90; // [rsp+170h] [rbp-A0h]
  __int64 *v91; // [rsp+178h] [rbp-98h]
  __int64 v92; // [rsp+180h] [rbp-90h]
  unsigned int *v93; // [rsp+190h] [rbp-80h] BYREF
  __int64 v94; // [rsp+198h] [rbp-78h] BYREF
  unsigned int v95; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v96; // [rsp+1A8h] [rbp-68h] BYREF
  unsigned int v97; // [rsp+1B0h] [rbp-60h]
  __int64 v98; // [rsp+1B8h] [rbp-58h] BYREF
  unsigned int v99; // [rsp+1C0h] [rbp-50h]
  __int64 v100; // [rsp+1C8h] [rbp-48h] BYREF
  unsigned int v101; // [rsp+1D0h] [rbp-40h]
  char v102; // [rsp+1D8h] [rbp-38h]

  if ( sub_AAF760(a2) )
    return (_QWORD *)sub_D970F0((__int64)a3);
  v6 = *(__int64 **)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *v6;
  v9 = 8 * v7;
  v10 = (8 * v7) >> 3;
  v11 = v10;
  if ( *(_WORD *)(*v6 + 24) )
  {
    v16 = (char *)v6 + v9;
    if ( v9 > 31 || v9 == 16 )
      goto LABEL_13;
    goto LABEL_17;
  }
  v12 = *(_QWORD *)(v8 + 32);
  v13 = *(_DWORD *)(v12 + 32);
  if ( v13 > 0x40 )
  {
    n = (8 * v7) >> 3;
    srca = (void *)(8 * v7);
    v46 = v6;
    v50 = *v6;
    v14 = sub_C444A0(v12 + 24);
    v8 = v50;
    v6 = v46;
    v9 = (signed __int64)srca;
    v10 = n;
    if ( v13 == v14 )
      goto LABEL_5;
LABEL_51:
    v93 = &v95;
    v94 = 0x400000000LL;
    if ( (unsigned __int64)v9 > 0x20 )
    {
      na = v9;
      srcb = v6;
      v47 = v8;
      v52 = v10;
      sub_C8D5F0((__int64)&v93, &v95, v10, 8u, (__int64)v6, v8);
      LODWORD(v10) = v52;
      v8 = v47;
      v6 = srcb;
      v9 = na;
      v27 = &v93[2 * (unsigned int)v94];
    }
    else
    {
      if ( !v9 )
      {
LABEL_53:
        LODWORD(v94) = v9 + v10;
        v51 = v8;
        v23 = sub_DA2C50((__int64)a3, *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8LL), 0, 0);
        v24 = (__int64 *)&v93;
        *(_QWORD *)v93 = v23;
        v25 = sub_DBFF60((__int64)a3, (unsigned int *)&v93, *(_QWORD *)(a1 + 48), *(_WORD *)(a1 + 28) & 1);
        if ( *((_WORD *)v25 + 12) == 8 )
        {
          sub_AB1F90((__int64)&v86, (__int64 *)a2, *(_QWORD *)(v51 + 32) + 24LL);
          v24 = (__int64 *)&v86;
          v22 = sub_DD1050(v25, &v86, a3);
          if ( (unsigned int)v89 > 0x40 && v88 )
            j_j___libc_free_0_0(v88);
          if ( (unsigned int)v87 > 0x40 && v86 )
            j_j___libc_free_0_0(v86);
        }
        else
        {
          v22 = sub_D970F0((__int64)a3);
        }
        if ( v93 != &v95 )
          _libc_free(v93, v24);
        return (_QWORD *)v22;
      }
      v27 = &v95;
    }
    v48 = v10;
    v53 = v8;
    memcpy(v27, v6, v9);
    LODWORD(v9) = v94;
    LODWORD(v10) = v48;
    v8 = v53;
    goto LABEL_53;
  }
  if ( *(_QWORD *)(v12 + 24) )
    goto LABEL_51;
LABEL_5:
  v15 = v10 >> 2;
  v16 = (char *)v6 + v9;
  if ( v15 > 0 )
  {
    v17 = &v6[4 * v15];
    while ( 1 )
    {
      if ( *(_WORD *)(v6[1] + 24) )
      {
        ++v6;
        goto LABEL_13;
      }
      if ( *(_WORD *)(v6[2] + 24) )
        break;
      if ( *(_WORD *)(v6[3] + 24) )
      {
        v6 += 3;
        goto LABEL_13;
      }
      v6 += 4;
      if ( v17 == v6 )
      {
        v11 = (v16 - (char *)v6) >> 3;
        if ( v16 - (char *)v6 != 16 )
          goto LABEL_17;
        goto LABEL_46;
      }
      if ( *(_WORD *)(*v6 + 24) )
        goto LABEL_13;
    }
    if ( v16 != (char *)(v6 + 2) )
      return (_QWORD *)sub_D970F0((__int64)a3);
    goto LABEL_19;
  }
  if ( v9 == 16 )
    goto LABEL_47;
LABEL_17:
  if ( v11 != 3 )
  {
    if ( v11 != 1 )
      goto LABEL_19;
LABEL_48:
    if ( !*(_WORD *)(*v6 + 24) )
      goto LABEL_19;
    goto LABEL_13;
  }
  if ( !*(_WORD *)(*v6 + 24) )
  {
    ++v6;
LABEL_46:
    if ( !*(_WORD *)(*v6 + 24) )
    {
LABEL_47:
      ++v6;
      goto LABEL_48;
    }
  }
LABEL_13:
  if ( v16 != (char *)v6 )
    return (_QWORD *)sub_D970F0((__int64)a3);
LABEL_19:
  v19 = sub_D95540(v8);
  LODWORD(v94) = sub_D97050((__int64)a3, v19);
  if ( (unsigned int)v94 > 0x40 )
    sub_C43690((__int64)&v93, 0, 0);
  else
    v93 = 0;
  v20 = sub_AB1B10(a2, (__int64)&v93);
  if ( (unsigned int)v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v20 )
  {
    v21 = *(_QWORD *)(a1 + 40);
    if ( v21 != 2 )
    {
      if ( v21 == 3 )
      {
        v56 = a1;
        v58 = 1;
        v57 = 0;
        v60 = 1;
        v59 = 0;
        v62 = 1;
        v61 = 0;
        v64 = 1;
        v63 = 0;
        sub_D92D30((__int64)&v93, a1);
        if ( v102 )
        {
          v90 = a3;
          v87 = &v55;
          v86 = (unsigned int *)&v63;
          v88 = &v57;
          v89 = &v59;
          v91 = &v56;
          v92 = a2;
          if ( v101 <= 0x40 )
          {
            v58 = v101;
            v57 = v100;
          }
          else
          {
            sub_C43990((__int64)&v57, (__int64)&v100);
          }
          if ( v60 <= 0x40 && v99 <= 0x40 )
          {
            v60 = v99;
            v59 = v98;
          }
          else
          {
            sub_C43990((__int64)&v59, (__int64)&v98);
          }
          if ( v62 <= 0x40 && v97 <= 0x40 )
          {
            v62 = v97;
            v61 = v96;
          }
          else
          {
            sub_C43990((__int64)&v61, (__int64)&v96);
          }
          if ( v64 <= 0x40 && v95 <= 0x40 )
          {
            v64 = v95;
            v63 = v94;
          }
          else
          {
            sub_C43990((__int64)&v63, (__int64)&v94);
          }
          v55 = (unsigned int)v93;
          sub_C44830((__int64)&v82, (_DWORD *)a2, v58);
          sub_C46F20((__int64)&v82, 1u);
          v66 = v83;
          v65 = v82;
          sub_C44830((__int64)&v67, (_DWORD *)(a2 + 16), v58);
          v83 = v66;
          if ( v66 > 0x40 )
            sub_C43780((__int64)&v82, &v65);
          else
            v82 = v65;
          sub_DD0560((__int64)&v78, (__int64)&v86, (__int64)&v82);
          if ( v83 > 0x40 && v82 )
            j_j___libc_free_0_0(v82);
          v76 = v68;
          if ( v68 > 0x40 )
            sub_C43780((__int64)&v75, (const void **)&v67);
          else
            v75 = v67;
          sub_DD0560((__int64)&v82, (__int64)&v86, (__int64)&v75);
          if ( v76 > 0x40 && v75 )
            j_j___libc_free_0_0(v75);
          if ( v81 && v85 )
          {
            v74 = 0;
            v40 = v55;
            if ( v84 )
            {
              sub_9865C0((__int64)v73, (__int64)&v82);
              v74 = 1;
            }
            v72 = 0;
            if ( v80 )
            {
              sub_9865C0((__int64)v71, (__int64)&v78);
              v72 = 1;
            }
            sub_D92C00((__int64)&v75, (__int64)v71, (__int64)v73);
            sub_D92430((__int64)v69, (__int64)&v75, v40);
            if ( v77 )
            {
              v77 = 0;
              sub_969240((__int64 *)&v75);
            }
            if ( v72 )
            {
              v72 = 0;
              sub_969240(v71);
            }
            if ( v74 )
            {
              v74 = 0;
              sub_969240(v73);
            }
          }
          else
          {
            v70 = 0;
          }
          if ( v84 )
          {
            v84 = 0;
            if ( v83 > 0x40 )
            {
              if ( v82 )
                j_j___libc_free_0_0(v82);
            }
          }
          if ( v80 )
          {
            v80 = 0;
            if ( v79 > 0x40 )
            {
              if ( v78 )
                j_j___libc_free_0_0(v78);
            }
          }
          if ( v68 > 0x40 && v67 )
            j_j___libc_free_0_0(v67);
          if ( v66 > 0x40 && v65 )
            j_j___libc_free_0_0(v65);
          if ( v102 )
          {
            v102 = 0;
            if ( v101 > 0x40 && v100 )
              j_j___libc_free_0_0(v100);
            if ( v99 > 0x40 && v98 )
              j_j___libc_free_0_0(v98);
            if ( v97 > 0x40 && v96 )
              j_j___libc_free_0_0(v96);
            if ( v95 > 0x40 && v94 )
              j_j___libc_free_0_0(v94);
          }
        }
        else
        {
          v70 = 0;
        }
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
        if ( v62 > 0x40 && v61 )
          j_j___libc_free_0_0(v61);
        if ( v60 > 0x40 && v59 )
          j_j___libc_free_0_0(v59);
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        if ( v70 )
        {
          v22 = (__int64)sub_DA26C0(a3, (__int64)v69);
          if ( v70 )
          {
            v70 = 0;
            sub_969240(v69);
          }
          return (_QWORD *)v22;
        }
      }
      return (_QWORD *)sub_D970F0((__int64)a3);
    }
    v28 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL) + 32LL);
    v29 = *(_DWORD *)(v28 + 32);
    v76 = v29;
    if ( v29 > 0x40 )
    {
      sub_C43780((__int64)&v75, (const void **)(v28 + 24));
      v29 = v76;
      if ( v76 > 0x40 )
      {
        v54 = v76;
        v49 = v76 - 1;
        src = (__int64 *)v75;
        if ( (*((_QWORD *)v75 + ((v76 - 1) >> 6)) & (1LL << ((unsigned __int8)v76 - 1))) != 0 )
        {
          v36 = sub_C44500((__int64)&v75);
          v37 = v54;
          v38 = v49;
          v39 = src;
        }
        else
        {
          v36 = sub_C444A0((__int64)&v75);
          v39 = src;
          v38 = v49;
          v37 = v54;
        }
        if ( v37 + 1 - v36 > 0x40 )
        {
          if ( !sub_986C60((__int64 *)&v75, v38) )
            goto LABEL_75;
          goto LABEL_93;
        }
        v30 = *v39;
LABEL_74:
        if ( v30 > 0 )
        {
LABEL_75:
          sub_9865C0((__int64)&v93, a2 + 16);
          sub_C46F20((__int64)&v93, 1u);
          v31 = v94;
          v79 = v94;
          v78 = (__int64)v93;
LABEL_76:
          LODWORD(v94) = v31;
          if ( v31 > 0x40 )
            sub_C43780((__int64)&v93, (const void **)&v78);
          else
            v93 = (unsigned int *)v78;
          sub_C45EE0((__int64)&v93, (__int64 *)&v75);
          LODWORD(v87) = v94;
          v86 = v93;
          LODWORD(v94) = 0;
          sub_C4A1D0((__int64)&v82, (__int64)&v86, (__int64)&v75);
          sub_969240((__int64 *)&v86);
          sub_969240((__int64 *)&v93);
          v32 = (__int64 *)sub_B2BE50(*a3);
          v33 = sub_ACCFD0(v32, (__int64)&v82);
          v34 = sub_DA2570((__int64)a3, v33);
          v35 = sub_DD0540(a1, (__int64)v34, a3);
          if ( sub_AB1B10(a2, v35[4] + 24LL) )
            v22 = sub_D970F0((__int64)a3);
          else
            v22 = (__int64)sub_DA2570((__int64)a3, v33);
          if ( v83 > 0x40 && v82 )
            j_j___libc_free_0_0(v82);
          sub_969240(&v78);
          sub_969240((__int64 *)&v75);
          return (_QWORD *)v22;
        }
LABEL_93:
        sub_9865C0((__int64)&v78, a2);
        v31 = v79;
        goto LABEL_76;
      }
    }
    else
    {
      v75 = *(void **)(v28 + 24);
    }
    if ( !v29 )
      goto LABEL_93;
    v30 = (__int64)((_QWORD)v75 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
    goto LABEL_74;
  }
  v26 = sub_D95540(**(_QWORD **)(a1 + 32));
  return sub_DA2C50((__int64)a3, v26, 0, 0);
}
