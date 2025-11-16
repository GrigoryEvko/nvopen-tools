// Function: sub_1A086C0
// Address: 0x1a086c0
//
__int64 __fastcall sub_1A086C0(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // rbx
  __int64 v5; // rcx
  __int64 *v6; // rax
  int v7; // r9d
  int v8; // r13d
  __int64 v9; // rcx
  unsigned int v10; // r15d
  __int64 *v11; // rsi
  __int64 *v12; // rsi
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  int v15; // eax
  _BYTE *v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r13
  int v20; // r8d
  __int64 v21; // r15
  _BYTE *v22; // r14
  __int64 v23; // rax
  char *v24; // r13
  char *v25; // r8
  char *v26; // r15
  __int64 v27; // rbx
  char *v28; // rax
  char *v29; // r14
  char *v30; // rcx
  char *v31; // rsi
  __int64 v32; // r15
  __int64 v33; // r13
  _QWORD *v34; // rbx
  bool v35; // cl
  __int64 v36; // r15
  unsigned int v37; // r15d
  __int64 v38; // rax
  char v39; // al
  __int64 v40; // rdx
  __int64 v41; // rcx
  bool v42; // cc
  char v43; // al
  __int64 v44; // rdi
  char v45; // al
  __int64 v46; // rdx
  __int64 v47; // rcx
  char v48; // al
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // r13
  __int64 *v53; // rax
  __int64 v54; // r15
  __int64 v55; // r8
  int v56; // r9d
  __int64 v57; // rax
  _QWORD *v58; // rax
  unsigned int v59; // r14d
  int v60; // eax
  __int64 v61; // r13
  _BYTE *v62; // rbx
  unsigned __int64 v63; // r12
  __int64 v64; // rdi
  __int64 v66; // r13
  int v67; // r8d
  int v68; // r9d
  __int64 v69; // r12
  __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // [rsp+8h] [rbp-258h]
  _DWORD *v73; // [rsp+10h] [rbp-250h]
  _QWORD *v74; // [rsp+28h] [rbp-238h]
  char v75; // [rsp+28h] [rbp-238h]
  char v76; // [rsp+28h] [rbp-238h]
  char v77; // [rsp+28h] [rbp-238h]
  __int64 v79; // [rsp+38h] [rbp-228h]
  __int64 v80; // [rsp+38h] [rbp-228h]
  __int64 *v81; // [rsp+38h] [rbp-228h]
  _BYTE *v82; // [rsp+38h] [rbp-228h]
  __int64 v83; // [rsp+38h] [rbp-228h]
  _QWORD *v84; // [rsp+48h] [rbp-218h] BYREF
  __int64 v85; // [rsp+50h] [rbp-210h] BYREF
  unsigned int v86; // [rsp+58h] [rbp-208h]
  __int64 v87; // [rsp+60h] [rbp-200h] BYREF
  __int64 v88; // [rsp+68h] [rbp-1F8h]
  const void *v89; // [rsp+70h] [rbp-1F0h] BYREF
  unsigned int v90; // [rsp+78h] [rbp-1E8h]
  int v91; // [rsp+80h] [rbp-1E0h]
  char v92; // [rsp+84h] [rbp-1DCh]
  void *src; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 v94; // [rsp+98h] [rbp-1C8h]
  _BYTE v95[64]; // [rsp+A0h] [rbp-1C0h] BYREF
  _BYTE *v96; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v97; // [rsp+E8h] [rbp-178h]
  _BYTE v98[368]; // [rsp+F0h] [rbp-170h] BYREF

  v4 = a3;
  v5 = *a3;
  v96 = v98;
  v97 = 0x800000000LL;
  v94 = 0x800000000LL;
  v6 = *(__int64 **)(v5 + 8);
  src = v95;
  v79 = v5;
  v72 = *v6;
  v86 = sub_16431D0(*v6);
  if ( v86 > 0x40 )
  {
    sub_16A4EF0((__int64)&v85, 0, 0);
    v8 = *((_DWORD *)v4 + 2);
    if ( v8 )
    {
      v9 = *v4;
      goto LABEL_3;
    }
LABEL_20:
    v18 = (unsigned int)v94;
    if ( (_DWORD)v97 )
    {
      v19 = 0;
      v20 = 40 * v97;
      v21 = 40LL * (unsigned int)v97;
      do
      {
        v22 = &v96[v19];
        if ( (unsigned int)v18 >= HIDWORD(v94) )
        {
          sub_16CD150((__int64)&src, v95, 0, 8, v20, v7);
          v18 = (unsigned int)v94;
        }
        v19 += 40;
        *((_QWORD *)src + v18) = v22;
        v18 = (unsigned int)(v94 + 1);
        LODWORD(v94) = v94 + 1;
      }
      while ( v21 != v19 );
    }
    v23 = 8 * v18;
    v24 = (char *)src;
    v25 = (char *)src + v23;
    if ( v23 )
    {
      v26 = (char *)src + v23;
      v74 = v4;
      v27 = v23 >> 3;
      do
      {
        v28 = (char *)sub_2207800(8 * v27, &unk_435FF63);
        v29 = v28;
        if ( v28 )
        {
          v30 = (char *)v27;
          v31 = v26;
          v32 = 8 * v27;
          v4 = v74;
          sub_1A00E20(v24, v31, v28, v30);
          goto LABEL_29;
        }
        v27 >>= 1;
      }
      while ( v27 );
      v4 = v74;
      v25 = v26;
    }
  }
  else
  {
    v8 = *((_DWORD *)v4 + 2);
    v9 = v79;
    v85 = 0;
    if ( v8 )
    {
LABEL_3:
      v10 = 0;
      while ( 1 )
      {
        v12 = *(__int64 **)(v9 + 16LL * v10 + 8);
        v13 = *((_BYTE *)v12 + 16);
        if ( v13 == 13 )
          break;
        v14 = *v12;
        if ( *(_BYTE *)(*v12 + 8) == 16 && v13 <= 0x10u )
        {
          v81 = *(__int64 **)(v9 + 16LL * v10 + 8);
          v38 = sub_15A1020(v81, (__int64)v12, v14, v9);
          v12 = v81;
          if ( v38 )
          {
            if ( *(_BYTE *)(v38 + 16) == 13 )
            {
              v11 = (__int64 *)(v38 + 24);
LABEL_5:
              if ( v86 > 0x40 )
                sub_16A8F00(&v85, v11);
              else
                v85 ^= *v11;
LABEL_7:
              if ( ++v10 == v8 )
                goto LABEL_20;
              goto LABEL_8;
            }
          }
        }
        sub_1A01620((__int64)&v87, (__int64)v12, v14, v9);
        v91 = sub_1A03A70(a1, v88);
        v15 = v97;
        if ( (unsigned int)v97 >= HIDWORD(v97) )
        {
          sub_1A01E90((__int64)&v96, 0);
          v15 = v97;
        }
        v16 = &v96[40 * v15];
        if ( v16 )
        {
          *(_QWORD *)v16 = v87;
          *((_QWORD *)v16 + 1) = v88;
          v17 = v90;
          *((_DWORD *)v16 + 6) = v90;
          if ( v17 > 0x40 )
          {
            v82 = v16;
            sub_16A4FD0((__int64)(v16 + 16), &v89);
            v16 = v82;
          }
          else
          {
            *((_QWORD *)v16 + 2) = v89;
          }
          *((_DWORD *)v16 + 8) = v91;
          v16[36] = v92;
          v15 = v97;
        }
        LODWORD(v97) = v15 + 1;
        if ( v90 <= 0x40 || !v89 )
          goto LABEL_7;
        j_j___libc_free_0_0(v89);
        if ( ++v10 == v8 )
          goto LABEL_20;
LABEL_8:
        v9 = *v4;
      }
      v11 = v12 + 3;
      goto LABEL_5;
    }
    v24 = v95;
    v25 = v95;
  }
  v32 = 0;
  v29 = 0;
  sub_1A00040(v24, v25);
LABEL_29:
  j_j___libc_free_0(v29, v32);
  if ( !(_DWORD)v97 )
    goto LABEL_68;
  v33 = 0;
  v73 = v4;
  v80 = 8LL * (unsigned int)v97;
  v34 = 0;
  v75 = 0;
  do
  {
    while ( 1 )
    {
      v37 = v86;
      if ( v86 > 0x40 )
        v35 = v37 == (unsigned int)sub_16A57B0((__int64)&v85);
      else
        v35 = v85 == 0;
      v36 = *(_QWORD *)((char *)src + v33);
      if ( v35 )
        goto LABEL_33;
      v39 = sub_1A084E0(a1, a2, *(unsigned __int64 **)((char *)src + v33), (__int64)&v85, (__int64 *)&v84);
      if ( !v39 )
        goto LABEL_33;
      if ( v84 )
        break;
      *(_QWORD *)v36 = 0;
      *(_QWORD *)(v36 + 8) = 0;
      v36 = (__int64)v34;
      v75 = v39;
LABEL_35:
      v34 = (_QWORD *)v36;
LABEL_36:
      v33 += 8;
      if ( v80 == v33 )
        goto LABEL_56;
    }
    v76 = v39;
    sub_1A01620((__int64)&v87, (__int64)v84, v40, v41);
    v42 = *(_DWORD *)(v36 + 24) <= 0x40u;
    v43 = v76;
    *(_QWORD *)v36 = v87;
    *(_QWORD *)(v36 + 8) = v88;
    if ( !v42 )
    {
      v44 = *(_QWORD *)(v36 + 16);
      if ( v44 )
      {
        j_j___libc_free_0_0(v44);
        v43 = v76;
      }
    }
    v75 = v43;
    *(_QWORD *)(v36 + 16) = v89;
    *(_DWORD *)(v36 + 24) = v90;
    *(_DWORD *)(v36 + 32) = v91;
    *(_BYTE *)(v36 + 36) = v92;
LABEL_33:
    if ( !v34 || *(_QWORD *)(v36 + 8) != v34[1] )
      goto LABEL_35;
    v45 = sub_1A07EC0(a1, a2, v36, (__int64)v34, (__int64)&v85, (__int64 *)&v84);
    if ( !v45 )
      goto LABEL_36;
    *v34 = 0;
    v34[1] = 0;
    v34 = v84;
    if ( !v84 )
    {
      *(_QWORD *)v36 = 0;
      *(_QWORD *)(v36 + 8) = 0;
      v75 = v45;
      goto LABEL_36;
    }
    v77 = v45;
    sub_1A01620((__int64)&v87, (__int64)v84, v46, v47);
    v42 = *(_DWORD *)(v36 + 24) <= 0x40u;
    v48 = v77;
    *(_QWORD *)v36 = v87;
    *(_QWORD *)(v36 + 8) = v88;
    if ( !v42 )
    {
      v49 = *(_QWORD *)(v36 + 16);
      if ( v49 )
      {
        j_j___libc_free_0_0(v49);
        v48 = v77;
      }
    }
    v75 = v48;
    v34 = (_QWORD *)v36;
    v33 += 8;
    *(_QWORD *)(v36 + 16) = v89;
    *(_DWORD *)(v36 + 24) = v90;
    *(_DWORD *)(v36 + 32) = v91;
    *(_BYTE *)(v36 + 36) = v92;
  }
  while ( v80 != v33 );
LABEL_56:
  if ( !v75 )
    goto LABEL_68;
  v50 = (unsigned int)v97;
  v73[2] = 0;
  if ( (_DWORD)v50 )
  {
    v51 = 0;
    v52 = 40 * v50;
    do
    {
      v53 = (__int64 *)&v96[v51];
      if ( *(_QWORD *)&v96[v51 + 8] )
      {
        v54 = *v53;
        v55 = (unsigned int)sub_1A03A70(a1, *v53);
        v57 = (unsigned int)v73[2];
        if ( (unsigned int)v57 >= v73[3] )
        {
          v83 = v55;
          sub_16CD150((__int64)v73, v73 + 4, 0, 16, v55, v56);
          v57 = (unsigned int)v73[2];
          v55 = v83;
        }
        v58 = (_QWORD *)(*(_QWORD *)v73 + 16 * v57);
        *v58 = v55;
        v58[1] = v54;
        ++v73[2];
      }
      v51 += 40;
    }
    while ( v51 != v52 );
  }
  v59 = v86;
  if ( v86 <= 0x40 )
  {
    if ( v85 )
      goto LABEL_91;
LABEL_66:
    v60 = v73[2];
    if ( v60 != 1 )
      goto LABEL_67;
LABEL_94:
    v61 = *(_QWORD *)(*(_QWORD *)v73 + 8LL);
  }
  else
  {
    if ( v59 == (unsigned int)sub_16A57B0((__int64)&v85) )
      goto LABEL_66;
LABEL_91:
    v66 = sub_15A1070(v72, (__int64)&v85);
    v69 = (unsigned int)sub_1A03A70(a1, v66);
    v70 = (unsigned int)v73[2];
    if ( (unsigned int)v70 >= v73[3] )
    {
      sub_16CD150((__int64)v73, v73 + 4, 0, 16, v67, v68);
      v70 = (unsigned int)v73[2];
    }
    v71 = (_QWORD *)(*(_QWORD *)v73 + 16 * v70);
    *v71 = v69;
    v71[1] = v66;
    v60 = v73[2] + 1;
    v73[2] = v60;
    if ( v60 == 1 )
      goto LABEL_94;
LABEL_67:
    if ( v60 )
LABEL_68:
      v61 = 0;
    else
      v61 = sub_15A1070(v72, (__int64)&v85);
  }
  if ( v86 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
  if ( src != v95 )
    _libc_free((unsigned __int64)src);
  v62 = v96;
  v63 = (unsigned __int64)&v96[40 * (unsigned int)v97];
  if ( v96 != (_BYTE *)v63 )
  {
    do
    {
      v63 -= 40LL;
      if ( *(_DWORD *)(v63 + 24) > 0x40u )
      {
        v64 = *(_QWORD *)(v63 + 16);
        if ( v64 )
          j_j___libc_free_0_0(v64);
      }
    }
    while ( v62 != (_BYTE *)v63 );
    v63 = (unsigned __int64)v96;
  }
  if ( (_BYTE *)v63 != v98 )
    _libc_free(v63);
  return v61;
}
