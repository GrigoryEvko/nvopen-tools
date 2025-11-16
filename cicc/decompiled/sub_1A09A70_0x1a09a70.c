// Function: sub_1A09A70
// Address: 0x1a09a70
//
__int64 *__fastcall sub_1A09A70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v12; // rax
  double v13; // xmm4_8
  double v14; // xmm5_8
  char v15; // al
  int v16; // r8d
  int v17; // r9d
  unsigned int v18; // r13d
  int v19; // r12d
  _BYTE *v20; // rsi
  char *v21; // r14
  int v22; // eax
  int v23; // r8d
  unsigned __int64 v24; // rbx
  int v25; // r9d
  __int64 v26; // rdx
  _BYTE *v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // r12
  __int64 v30; // r14
  unsigned int v31; // ecx
  unsigned __int64 v32; // rsi
  unsigned int v33; // ecx
  const void *v34; // r15
  bool v35; // cc
  bool v36; // al
  bool v37; // r13
  _BYTE *v38; // rdi
  __int64 v39; // r13
  char v40; // al
  __int64 *v41; // rsi
  unsigned __int64 *v42; // r15
  __int64 *v43; // r13
  _BYTE *v44; // rbx
  unsigned __int64 v45; // r12
  __int64 v46; // rdi
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // r13
  int v51; // edx
  _BYTE *v52; // rax
  int v53; // eax
  char v54; // r14
  __int64 v55; // r12
  __int64 v56; // r15
  int v57; // eax
  _BYTE *v58; // rdx
  _BYTE *v59; // rsi
  int v60; // edx
  _BYTE *v61; // rax
  _BYTE *v62; // rsi
  __int64 v63; // [rsp+0h] [rbp-200h]
  __int64 *v65; // [rsp+18h] [rbp-1E8h]
  __int64 v66; // [rsp+30h] [rbp-1D0h]
  unsigned int v67; // [rsp+3Ch] [rbp-1C4h]
  int v68; // [rsp+3Ch] [rbp-1C4h]
  void *v69; // [rsp+40h] [rbp-1C0h] BYREF
  unsigned int v70; // [rsp+48h] [rbp-1B8h]
  char *v71; // [rsp+50h] [rbp-1B0h] BYREF
  unsigned __int64 *v72; // [rsp+58h] [rbp-1A8h] BYREF
  __int64 v73; // [rsp+60h] [rbp-1A0h]
  _BYTE *v74; // [rsp+70h] [rbp-190h] BYREF
  __int64 v75; // [rsp+78h] [rbp-188h]
  _BYTE v76[128]; // [rsp+80h] [rbp-180h] BYREF
  _BYTE *v77; // [rsp+100h] [rbp-100h] BYREF
  __int64 v78; // [rsp+108h] [rbp-F8h]
  _BYTE v79[240]; // [rsp+110h] [rbp-F0h] BYREF

  v12 = (__int64 *)sub_19FEFC0(a2, 15, 16);
  v65 = v12;
  if ( !v12 )
    return 0;
  v77 = v79;
  v78 = 0x800000000LL;
  v15 = sub_1A04A80(v12, (__int64)&v77, a4, a5, a6, a7, v13, v14, a10, a11);
  v18 = v78;
  *(_BYTE *)(a1 + 752) |= v15;
  v74 = v76;
  v75 = 0x800000000LL;
  if ( v18 > 8 )
  {
    sub_16CD150((__int64)&v74, v76, v18, 16, v16, v17);
    v18 = v78;
    if ( !(_DWORD)v78 )
      goto LABEL_22;
  }
  else if ( !v18 )
  {
LABEL_48:
    v43 = 0;
    sub_1A08FD0(a1, (__int64)v65, (__int64 *)&v74);
    goto LABEL_49;
  }
  v63 = a3;
  v19 = 0;
  do
  {
    v20 = &v77[24 * v19];
    v21 = *(char **)v20;
    v71 = *(char **)v20;
    LODWORD(v73) = *((_DWORD *)v20 + 4);
    if ( (unsigned int)v73 > 0x40 )
    {
      sub_16A4FD0((__int64)&v72, (const void **)v20 + 1);
      v21 = v71;
    }
    else
    {
      v72 = (unsigned __int64 *)*((_QWORD *)v20 + 1);
    }
    v22 = sub_1A03A70(a1, (__int64)v21);
    v24 = (unsigned __int64)v72;
    v25 = v22;
    if ( (unsigned int)v73 > 0x40 )
      v24 = *v72;
    v26 = (unsigned int)v75;
    if ( v24 > HIDWORD(v75) - (unsigned __int64)(unsigned int)v75 )
    {
      v68 = v22;
      sub_16CD150((__int64)&v74, v76, v24 + (unsigned int)v75, 16, v23, v22);
      v26 = (unsigned int)v75;
      v25 = v68;
    }
    v27 = &v74[16 * v26];
    if ( v24 )
    {
      v28 = v24;
      do
      {
        if ( v27 )
        {
          *(_DWORD *)v27 = v25;
          *((_QWORD *)v27 + 1) = v21;
        }
        v27 += 16;
        --v28;
      }
      while ( v28 );
      LODWORD(v26) = v75;
    }
    LODWORD(v75) = v24 + v26;
    if ( (unsigned int)v73 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
    ++v19;
  }
  while ( v19 != v18 );
  a3 = v63;
LABEL_22:
  if ( !(_DWORD)v75 )
    goto LABEL_48;
  v66 = a3 + 24;
  v29 = 16LL * (unsigned int)v75;
  v30 = 0;
  while ( 1 )
  {
    v38 = &v74[v30];
    v39 = *(_QWORD *)&v74[v30 + 8];
    if ( v39 == a3 )
      break;
    v40 = *(_BYTE *)(a3 + 16);
    if ( v40 == 13 )
    {
      if ( *(_BYTE *)(v39 + 16) != 13 )
        goto LABEL_36;
      v31 = *(_DWORD *)(v39 + 32);
      v70 = v31;
      if ( v31 <= 0x40 )
      {
        v32 = *(_QWORD *)(v39 + 24);
LABEL_27:
        v69 = (void *)(~v32 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31));
        goto LABEL_28;
      }
      sub_16A4FD0((__int64)&v69, (const void **)(v39 + 24));
      LOBYTE(v31) = v70;
      if ( v70 <= 0x40 )
      {
        v32 = (unsigned __int64)v69;
        goto LABEL_27;
      }
      sub_16A8F40((__int64 *)&v69);
LABEL_28:
      sub_16A7400((__int64)&v69);
      v33 = v70;
      v34 = v69;
      v70 = 0;
      v35 = *(_DWORD *)(a3 + 32) <= 0x40u;
      LODWORD(v72) = v33;
      v71 = (char *)v69;
      if ( v35 )
      {
        v37 = *(_QWORD *)(a3 + 24) == (_QWORD)v69;
      }
      else
      {
        v67 = v33;
        v36 = sub_16A5220(v66, (const void **)&v71);
        v33 = v67;
        v37 = v36;
      }
      if ( v33 > 0x40 )
      {
        if ( v34 )
        {
          j_j___libc_free_0_0(v34);
          if ( v70 > 0x40 )
          {
            if ( v69 )
              j_j___libc_free_0_0(v69);
          }
        }
      }
      if ( v37 )
      {
        v56 = v30;
        v54 = v37;
        v57 = v75;
        v58 = &v74[16 * (unsigned int)v75];
        v59 = &v74[v56 + 16];
        if ( v58 != v59 )
        {
          memmove(&v74[v56], v59, v58 - v59);
          v57 = v75;
        }
        v53 = v57 - 1;
        LODWORD(v75) = v53;
        goto LABEL_73;
      }
LABEL_36:
      v30 += 16;
      if ( v29 == v30 )
        goto LABEL_48;
    }
    else
    {
      if ( v40 != 14 || *(_BYTE *)(v39 + 16) != 14 )
        goto LABEL_36;
      v41 = (__int64 *)(v39 + 32);
      v42 = (unsigned __int64 *)sub_16982C0();
      if ( *(unsigned __int64 **)(v39 + 32) == v42 )
        sub_169C6E0(&v72, (__int64)v41);
      else
        sub_16986C0(&v72, v41);
      if ( v72 == v42 )
        sub_169C8D0((__int64)&v72, *(double *)a4.m128_u64, a5, a6);
      else
        sub_1699490((__int64)&v72);
      if ( (unsigned int)sub_14A9E40(v66, (__int64)&v71) == 1 )
      {
        v60 = v75;
        v61 = &v74[16 * (unsigned int)v75];
        v62 = &v74[v30 + 16];
        if ( v61 != v62 )
        {
          memmove(&v74[v30], v62, v61 - v62);
          v60 = v75;
        }
        v54 = 1;
        LODWORD(v75) = v60 - 1;
        sub_127D120(&v72);
        v53 = v75;
        goto LABEL_73;
      }
      if ( v42 == v72 )
      {
        v48 = v73;
        if ( v73 )
        {
          v49 = 32LL * *(_QWORD *)(v73 - 8);
          v50 = v73 + v49;
          if ( v73 != v73 + v49 )
          {
            do
            {
              v50 -= 32;
              sub_127D120((_QWORD *)(v50 + 8));
            }
            while ( v48 != v50 );
          }
          j_j_j___libc_free_0_0(v48 - 8);
        }
        goto LABEL_36;
      }
      v30 += 16;
      sub_1698460((__int64)&v72);
      if ( v29 == v30 )
        goto LABEL_48;
    }
  }
  v51 = v75;
  v52 = &v74[16 * (unsigned int)v75];
  if ( v52 != v38 + 16 )
  {
    memmove(v38, v38 + 16, v52 - (v38 + 16));
    v51 = v75;
  }
  v53 = v51 - 1;
  v54 = 0;
  LODWORD(v75) = v51 - 1;
LABEL_73:
  v55 = v65[4];
  if ( v53 == 1 )
  {
    v71 = (char *)v65;
    sub_1A062A0(a1 + 64, &v71);
    v43 = (__int64 *)*((_QWORD *)v74 + 1);
  }
  else
  {
    v43 = v65;
    sub_1A08FD0(a1, (__int64)v65, (__int64 *)&v74);
  }
  if ( v54 )
  {
    if ( v55 )
      v55 -= 24;
    v71 = "neg";
    LOWORD(v73) = 259;
    v43 = (__int64 *)sub_19FE020(v43, (__int64)&v71, v55, (__int64)v65);
  }
LABEL_49:
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  v44 = v77;
  v45 = (unsigned __int64)&v77[24 * (unsigned int)v78];
  if ( v77 != (_BYTE *)v45 )
  {
    do
    {
      v45 -= 24LL;
      if ( *(_DWORD *)(v45 + 16) > 0x40u )
      {
        v46 = *(_QWORD *)(v45 + 8);
        if ( v46 )
          j_j___libc_free_0_0(v46);
      }
    }
    while ( v44 != (_BYTE *)v45 );
    v45 = (unsigned __int64)v77;
  }
  if ( (_BYTE *)v45 != v79 )
    _libc_free(v45);
  return v43;
}
