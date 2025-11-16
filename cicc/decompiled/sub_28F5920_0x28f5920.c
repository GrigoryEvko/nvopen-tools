// Function: sub_28F5920
// Address: 0x28f5920
//
__int64 __fastcall sub_28F5920(__int64 a1, __int64 a2, _QWORD *a3)
{
  _DWORD *v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  unsigned __int64 v9; // r9
  int v10; // r13d
  __int64 v11; // rcx
  unsigned int v12; // r14d
  __int64 v13; // r15
  __int64 *v14; // rsi
  _BYTE *v15; // rax
  int v16; // eax
  __int64 v17; // rsi
  char *v18; // rcx
  __int64 *v19; // rdi
  int v20; // eax
  __int64 *v21; // r15
  unsigned int v22; // eax
  __int64 *v23; // r13
  __int64 *v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  char *v27; // r13
  char *v28; // r8
  char *v29; // r14
  __int64 v30; // rbx
  char *v31; // rax
  unsigned __int64 v32; // r15
  char *v33; // rcx
  __int64 v34; // r12
  char *v35; // rbx
  bool v36; // dl
  __int64 v37; // r15
  unsigned int v38; // r15d
  char v39; // al
  bool v40; // cc
  char v41; // al
  unsigned __int64 v42; // rdi
  __int64 v43; // rax
  __int16 v44; // ax
  char v45; // al
  unsigned __int64 v46; // rdi
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 *v49; // r15
  __int64 v50; // rax
  unsigned __int64 v51; // r14
  unsigned __int64 v52; // r13
  __int64 v53; // r8
  unsigned __int64 v54; // r9
  __int64 v55; // rax
  unsigned __int64 *v56; // rax
  unsigned int v57; // r14d
  int v58; // eax
  __int64 v59; // r13
  __int64 *v60; // rbx
  unsigned __int64 v61; // r12
  unsigned __int64 v62; // rdi
  __int64 *v64; // r15
  _BYTE *v65; // r13
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // r12
  __int64 v69; // rax
  _QWORD *v70; // rax
  unsigned __int64 v71; // [rsp+8h] [rbp-278h]
  __int64 v72; // [rsp+18h] [rbp-268h]
  _DWORD *v73; // [rsp+20h] [rbp-260h]
  __int64 v74; // [rsp+38h] [rbp-248h]
  char v75; // [rsp+40h] [rbp-240h]
  char v76; // [rsp+40h] [rbp-240h]
  char v77; // [rsp+40h] [rbp-240h]
  char *v78; // [rsp+40h] [rbp-240h]
  _DWORD *v80; // [rsp+50h] [rbp-230h]
  __int64 v81; // [rsp+50h] [rbp-230h]
  unsigned __int64 v82; // [rsp+50h] [rbp-230h]
  __int64 v83; // [rsp+58h] [rbp-228h]
  __int64 v84; // [rsp+58h] [rbp-228h]
  __int64 *v85; // [rsp+58h] [rbp-228h]
  char *v86; // [rsp+68h] [rbp-218h] BYREF
  unsigned __int64 v87; // [rsp+70h] [rbp-210h] BYREF
  unsigned int v88; // [rsp+78h] [rbp-208h]
  __int64 v89; // [rsp+80h] [rbp-200h] BYREF
  _BYTE *v90; // [rsp+88h] [rbp-1F8h]
  unsigned __int64 v91; // [rsp+90h] [rbp-1F0h]
  unsigned int v92; // [rsp+98h] [rbp-1E8h]
  int v93; // [rsp+A0h] [rbp-1E0h]
  char v94; // [rsp+A4h] [rbp-1DCh]
  void *src; // [rsp+B0h] [rbp-1D0h] BYREF
  __int64 v96; // [rsp+B8h] [rbp-1C8h]
  _BYTE v97[64]; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 *v98; // [rsp+100h] [rbp-180h] BYREF
  __int64 v99; // [rsp+108h] [rbp-178h]
  _BYTE v100[368]; // [rsp+110h] [rbp-170h] BYREF

  v4 = a3;
  v5 = *a3;
  v98 = (__int64 *)v100;
  v99 = 0x800000000LL;
  v96 = 0x800000000LL;
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 8LL);
  src = v97;
  v83 = v5;
  v72 = v6;
  v88 = sub_BCB060(v6);
  if ( v88 > 0x40 )
  {
    sub_C43690((__int64)&v87, 0, 0);
    v10 = v4[2];
    if ( v10 )
    {
      v11 = *(_QWORD *)v4;
      goto LABEL_3;
    }
LABEL_21:
    v23 = v98;
    v24 = &v98[5 * (unsigned int)v99];
    v25 = (unsigned int)v96;
    if ( v98 != v24 )
    {
      do
      {
        if ( v25 + 1 > (unsigned __int64)HIDWORD(v96) )
        {
          sub_C8D5F0((__int64)&src, v97, v25 + 1, 8u, v8, v9);
          v25 = (unsigned int)v96;
        }
        *((_QWORD *)src + v25) = v23;
        v23 += 5;
        v25 = (unsigned int)(v96 + 1);
        LODWORD(v96) = v96 + 1;
      }
      while ( v24 != v23 );
    }
    v26 = 8 * v25;
    v27 = (char *)src;
    v28 = (char *)src + v26;
    if ( v26 )
    {
      v29 = (char *)src + v26;
      v80 = v4;
      v30 = v26 >> 3;
      do
      {
        v31 = (char *)sub_2207800(8 * v30);
        v32 = (unsigned __int64)v31;
        if ( v31 )
        {
          v33 = (char *)v30;
          v4 = v80;
          sub_28EC030(v27, v29, v31, v33);
          goto LABEL_29;
        }
        v30 >>= 1;
      }
      while ( v30 );
      v4 = v80;
      v28 = v29;
    }
  }
  else
  {
    v10 = v4[2];
    v11 = v83;
    v87 = 0;
    if ( v10 )
    {
LABEL_3:
      v12 = 0;
      while ( 1 )
      {
        v13 = *(_QWORD *)(v11 + 16LL * v12 + 8);
        v14 = (__int64 *)(v13 + 24);
        if ( *(_BYTE *)v13 == 17 )
          break;
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v13 <= 0x15u )
        {
          v15 = sub_AD7630(v13, 0, v7);
          if ( v15 )
          {
            v14 = (__int64 *)(v15 + 24);
            if ( *v15 == 17 )
              break;
          }
        }
        sub_28ED4B0((__int64)&v89, (char *)v13);
        v16 = sub_28EF780(a1, v90);
        v17 = (unsigned int)v99;
        v18 = (char *)&v89;
        v93 = v16;
        v19 = v98;
        v9 = (unsigned int)v99 + 1LL;
        v20 = v99;
        if ( v9 > HIDWORD(v99) )
        {
          v64 = v98;
          if ( v98 > &v89 || &v89 >= &v98[5 * (unsigned int)v99] )
          {
            sub_28EF140((__int64)&v98, (unsigned int)v99 + 1LL, v7, (__int64)&v89, v8, v9);
            v17 = (unsigned int)v99;
            v19 = v98;
            v18 = (char *)&v89;
            v20 = v99;
          }
          else
          {
            sub_28EF140((__int64)&v98, (unsigned int)v99 + 1LL, v7, (__int64)&v89, v8, v9);
            v19 = v98;
            v17 = (unsigned int)v99;
            v18 = (char *)v98 + (char *)&v89 - (char *)v64;
            v20 = v99;
          }
        }
        v21 = &v19[5 * v17];
        if ( v21 )
        {
          *v21 = *(_QWORD *)v18;
          v21[1] = *((_QWORD *)v18 + 1);
          v22 = *((_DWORD *)v18 + 6);
          *((_DWORD *)v21 + 6) = v22;
          if ( v22 > 0x40 )
          {
            v78 = v18;
            sub_C43780((__int64)(v21 + 2), (const void **)v18 + 2);
            v18 = v78;
          }
          else
          {
            v21[2] = *((_QWORD *)v18 + 2);
          }
          *((_DWORD *)v21 + 8) = *((_DWORD *)v18 + 8);
          *((_BYTE *)v21 + 36) = v18[36];
          v20 = v99;
        }
        LODWORD(v99) = v20 + 1;
        if ( v92 <= 0x40 || !v91 )
          goto LABEL_6;
        j_j___libc_free_0_0(v91);
        if ( ++v12 == v10 )
          goto LABEL_21;
LABEL_7:
        v11 = *(_QWORD *)v4;
      }
      if ( v88 > 0x40 )
        sub_C43C10(&v87, v14);
      else
        v87 ^= *v14;
LABEL_6:
      if ( ++v12 == v10 )
        goto LABEL_21;
      goto LABEL_7;
    }
    v27 = v97;
    v28 = v97;
  }
  v32 = 0;
  sub_28EAE60(v27, v28);
LABEL_29:
  j_j___libc_free_0(v32);
  if ( !(_DWORD)v99 )
    goto LABEL_64;
  v81 = a1;
  v73 = v4;
  v34 = 0;
  v35 = 0;
  v84 = 8LL * (unsigned int)v99;
  v75 = 0;
  do
  {
    while ( 1 )
    {
      v38 = v88;
      if ( v88 > 0x40 )
        v36 = v38 == (unsigned int)sub_C444A0((__int64)&v87);
      else
        v36 = v87 == 0;
      v37 = *(_QWORD *)((char *)src + v34);
      if ( v36 )
        goto LABEL_33;
      v39 = sub_28F56C0(v81, a2 + 24, 0, *(__int64 **)((char *)src + v34), (__int64)&v87, (__int64 *)&v86);
      if ( !v39 )
        goto LABEL_33;
      if ( v86 )
        break;
      *(_QWORD *)v37 = 0;
      *(_QWORD *)(v37 + 8) = 0;
      v37 = (__int64)v35;
      v75 = v39;
LABEL_35:
      v35 = (char *)v37;
LABEL_36:
      v34 += 8;
      if ( v84 == v34 )
        goto LABEL_52;
    }
    v76 = v39;
    sub_28ED4B0((__int64)&v89, v86);
    v40 = *(_DWORD *)(v37 + 24) <= 0x40u;
    v41 = v76;
    *(_QWORD *)v37 = v89;
    *(_QWORD *)(v37 + 8) = v90;
    if ( !v40 )
    {
      v42 = *(_QWORD *)(v37 + 16);
      if ( v42 )
      {
        j_j___libc_free_0_0(v42);
        v41 = v76;
      }
    }
    v75 = v41;
    *(_QWORD *)(v37 + 16) = v91;
    *(_DWORD *)(v37 + 24) = v92;
    *(_DWORD *)(v37 + 32) = v93;
    *(_BYTE *)(v37 + 36) = v94;
LABEL_33:
    if ( !v35 || *(_QWORD *)(v37 + 8) != *((_QWORD *)v35 + 1) )
      goto LABEL_35;
    v43 = v74;
    LOWORD(v43) = 0;
    v74 = v43;
    v44 = (unsigned __int8)sub_28F4ED0(v81, a2 + 24, 0, v37, (__int64)v35, (__int64)&v87, (__int64 *)&v86);
    if ( !(_BYTE)v44 )
      goto LABEL_36;
    *(_QWORD *)v35 = 0;
    *((_QWORD *)v35 + 1) = 0;
    v35 = v86;
    if ( !v86 )
    {
      *(_QWORD *)v37 = 0;
      *(_QWORD *)(v37 + 8) = 0;
      v75 = v44;
      goto LABEL_36;
    }
    v77 = v44;
    sub_28ED4B0((__int64)&v89, v86);
    v40 = *(_DWORD *)(v37 + 24) <= 0x40u;
    v45 = v77;
    *(_QWORD *)v37 = v89;
    *(_QWORD *)(v37 + 8) = v90;
    if ( !v40 )
    {
      v46 = *(_QWORD *)(v37 + 16);
      if ( v46 )
      {
        j_j___libc_free_0_0(v46);
        v45 = v77;
      }
    }
    v75 = v45;
    v35 = (char *)v37;
    v34 += 8;
    *(_QWORD *)(v37 + 16) = v91;
    *(_DWORD *)(v37 + 24) = v92;
    *(_DWORD *)(v37 + 32) = v93;
    *(_BYTE *)(v37 + 36) = v94;
  }
  while ( v84 != v34 );
LABEL_52:
  v47 = v81;
  if ( !v75 )
    goto LABEL_64;
  v48 = (unsigned int)v99;
  v49 = v98;
  v73[2] = 0;
  v50 = 5 * v48;
  if ( v49 != &v49[v50] )
  {
    v85 = &v49[v50];
    v51 = v71;
    do
    {
      if ( v49[1] )
      {
        v52 = *v49;
        v54 = (unsigned int)sub_28EF780(v47, (_BYTE *)*v49) | v51 & 0xFFFFFFFF00000000LL;
        v55 = (unsigned int)v73[2];
        v51 = v54;
        if ( v55 + 1 > (unsigned __int64)(unsigned int)v73[3] )
        {
          v82 = v54;
          sub_C8D5F0((__int64)v73, v73 + 4, v55 + 1, 0x10u, v53, v54);
          v55 = (unsigned int)v73[2];
          v54 = v82;
        }
        v56 = (unsigned __int64 *)(*(_QWORD *)v73 + 16 * v55);
        *v56 = v54;
        v56[1] = v52;
        ++v73[2];
      }
      v49 += 5;
    }
    while ( v85 != v49 );
  }
  v57 = v88;
  if ( v88 <= 0x40 )
  {
    if ( v87 )
      goto LABEL_91;
LABEL_62:
    v58 = v73[2];
    if ( v58 != 1 )
      goto LABEL_63;
LABEL_94:
    v59 = *(_QWORD *)(*(_QWORD *)v73 + 8LL);
  }
  else
  {
    if ( v57 == (unsigned int)sub_C444A0((__int64)&v87) )
      goto LABEL_62;
LABEL_91:
    v65 = (_BYTE *)sub_AD8D80(v72, (__int64)&v87);
    v68 = (unsigned int)sub_28EF780(v47, v65);
    v69 = (unsigned int)v73[2];
    if ( v69 + 1 > (unsigned __int64)(unsigned int)v73[3] )
    {
      sub_C8D5F0((__int64)v73, v73 + 4, v69 + 1, 0x10u, v66, v67);
      v69 = (unsigned int)v73[2];
    }
    v70 = (_QWORD *)(*(_QWORD *)v73 + 16 * v69);
    *v70 = v68;
    v70[1] = v65;
    v58 = v73[2] + 1;
    v73[2] = v58;
    if ( v58 == 1 )
      goto LABEL_94;
LABEL_63:
    if ( v58 )
LABEL_64:
      v59 = 0;
    else
      v59 = sub_AD8D80(v72, (__int64)&v87);
  }
  if ( v88 > 0x40 && v87 )
    j_j___libc_free_0_0(v87);
  if ( src != v97 )
    _libc_free((unsigned __int64)src);
  v60 = v98;
  v61 = (unsigned __int64)&v98[5 * (unsigned int)v99];
  if ( v98 != (__int64 *)v61 )
  {
    do
    {
      v61 -= 40LL;
      if ( *(_DWORD *)(v61 + 24) > 0x40u )
      {
        v62 = *(_QWORD *)(v61 + 16);
        if ( v62 )
          j_j___libc_free_0_0(v62);
      }
    }
    while ( v60 != (__int64 *)v61 );
    v61 = (unsigned __int64)v98;
  }
  if ( (_BYTE *)v61 != v100 )
    _libc_free(v61);
  return v59;
}
