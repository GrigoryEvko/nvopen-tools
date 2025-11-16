// Function: sub_391EDB0
// Address: 0x391edb0
//
__int64 __fastcall sub_391EDB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r14
  char v20; // al
  __int64 v21; // rcx
  int v22; // r9d
  _DWORD *v23; // r15
  int *v24; // r8
  _QWORD *v25; // rdx
  char v26; // r13
  unsigned int v27; // esi
  int v28; // r14d
  __int64 v29; // r8
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // rdx
  unsigned int v34; // r12d
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // r14
  unsigned __int64 v39; // r14
  __int64 v40; // rdx
  int v41; // esi
  int v42; // esi
  __int64 v43; // r8
  unsigned int v44; // ecx
  int v45; // edx
  __int64 v46; // rdi
  int v47; // r10d
  __int64 *v48; // r9
  unsigned int v49; // esi
  int v50; // eax
  int v51; // eax
  int v52; // r9d
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rcx
  int v56; // r8d
  int v57; // r9d
  int v58; // r10d
  __int64 *v59; // r11
  int v60; // edx
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // rdi
  int v64; // r9d
  unsigned int v65; // r13d
  __int64 *v66; // r8
  __int64 v67; // rsi
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rcx
  __int64 v71; // rax
  __int64 v72; // rdx
  unsigned __int64 v73; // r13
  unsigned __int64 v74; // rbx
  __int64 v75; // r13
  char **v76; // r12
  int v77; // eax
  unsigned __int64 v78; // rbx
  unsigned __int64 v79; // rdi
  unsigned __int64 v80; // rdi
  __int64 v81; // [rsp+28h] [rbp-148h]
  int v83; // [rsp+38h] [rbp-138h]
  _DWORD *v84; // [rsp+58h] [rbp-118h] BYREF
  int v85; // [rsp+60h] [rbp-110h]
  char *v86; // [rsp+68h] [rbp-108h] BYREF
  __int64 v87; // [rsp+70h] [rbp-100h]
  char v88; // [rsp+78h] [rbp-F8h] BYREF
  _BYTE *v89; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v90; // [rsp+88h] [rbp-E8h]
  _BYTE v91[16]; // [rsp+90h] [rbp-E0h] BYREF
  int v92; // [rsp+A0h] [rbp-D0h]
  char *v93; // [rsp+A8h] [rbp-C8h] BYREF
  __int64 v94; // [rsp+B0h] [rbp-C0h]
  char v95; // [rsp+B8h] [rbp-B8h] BYREF
  char *v96; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+C8h] [rbp-A8h]
  _BYTE v98[16]; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 v99; // [rsp+E0h] [rbp-90h]
  int v100; // [rsp+F0h] [rbp-80h] BYREF
  char *v101[2]; // [rsp+F8h] [rbp-78h] BYREF
  char v102; // [rsp+108h] [rbp-68h] BYREF
  char *v103[2]; // [rsp+110h] [rbp-60h] BYREF
  _BYTE v104[16]; // [rsp+120h] [rbp-50h] BYREF
  int v105; // [rsp+130h] [rbp-40h]

  v6 = a2;
  v7 = a2;
  v8 = a1;
  v86 = &v88;
  v87 = 0x100000000LL;
  v89 = v91;
  v90 = 0x400000000LL;
  v9 = *(_BYTE *)(a2 + 9);
  v85 = 0;
  if ( (v9 & 0xC) == 8 )
  {
    v10 = *(_QWORD *)(a2 + 24);
    *(_BYTE *)(a2 + 8) |= 4u;
    v6 = *(_QWORD *)(v10 + 24);
  }
  sub_39199E0((__int64)&v86, v6 + 72, a3, a4, a5, a6);
  sub_39199E0((__int64)&v89, v6 + 96, v11, v12, v13, v14);
  v19 = *(unsigned int *)(a1 + 352);
  v92 = v85;
  v93 = &v95;
  v94 = 0x100000000LL;
  if ( (_DWORD)v87 )
    sub_39199E0((__int64)&v93, (__int64)&v86, v15, v16, v17, v18);
  v96 = v98;
  v97 = 0x400000000LL;
  if ( (_DWORD)v90 )
    sub_39199E0((__int64)&v96, (__int64)&v89, v15, v16, v17, v18);
  v99 = v19;
  v100 = v92;
  v101[0] = &v102;
  v101[1] = (char *)0x100000000LL;
  if ( (_DWORD)v94 )
    sub_3919AC0((__int64)v101, &v93, v15, v16, v17, v18);
  v103[0] = v104;
  v103[1] = (char *)0x400000000LL;
  if ( (_DWORD)v97 )
    sub_3919AC0((__int64)v103, &v96, v15, v16, v17, v97);
  v105 = v99;
  v20 = sub_391AD00(a1 + 312, &v100, &v84);
  v23 = v84;
  v24 = &v100;
  v25 = &v84;
  if ( v20 )
  {
    v26 = 0;
    goto LABEL_13;
  }
  v49 = *(_DWORD *)(a1 + 336);
  v50 = *(_DWORD *)(a1 + 328);
  ++*(_QWORD *)(a1 + 312);
  v51 = v50 + 1;
  v52 = 2 * v49;
  if ( 4 * v51 >= 3 * v49 )
  {
    v49 *= 2;
  }
  else
  {
    v53 = v49 - *(_DWORD *)(a1 + 332) - v51;
    if ( (unsigned int)v53 > v49 >> 3 )
      goto LABEL_48;
  }
  sub_391AF60(a1 + 312, v49);
  sub_391AD00(a1 + 312, &v100, &v84);
  v23 = v84;
  v51 = *(_DWORD *)(a1 + 328) + 1;
LABEL_48:
  *(_DWORD *)(a1 + 328) = v51;
  if ( *v23 != 1 || (LODWORD(v24) = v23[4], (_DWORD)v24) || v23[10] )
    --*(_DWORD *)(a1 + 332);
  v26 = 1;
  *v23 = v100;
  sub_3919AC0((__int64)(v23 + 2), v101, (__int64)v25, v53, (int)v24, v52);
  sub_3919AC0((__int64)(v23 + 8), v103, v54, v55, v56, v57);
  v23[16] = v105;
LABEL_13:
  if ( v103[0] != v104 )
    _libc_free((unsigned __int64)v103[0]);
  if ( v101[0] != &v102 )
    _libc_free((unsigned __int64)v101[0]);
  if ( v96 != v98 )
    _libc_free((unsigned __int64)v96);
  if ( v93 != &v95 )
    _libc_free((unsigned __int64)v93);
  if ( !v26 )
  {
    v27 = *(_DWORD *)(a1 + 120);
    v28 = v23[16];
    v29 = a1 + 96;
    if ( v27 )
      goto LABEL_23;
LABEL_38:
    ++*(_QWORD *)(v8 + 96);
    goto LABEL_39;
  }
  v36 = *(unsigned int *)(a1 + 352);
  v37 = *(unsigned int *)(a1 + 356);
  if ( (unsigned int)v36 >= (unsigned int)v37 )
  {
    v68 = ((((unsigned __int64)(v37 + 2) >> 1) | (v37 + 2)) >> 2) | ((unsigned __int64)(v37 + 2) >> 1) | (v37 + 2);
    v69 = (((v68 >> 4) | v68) >> 8) | (v68 >> 4) | v68;
    v70 = (v69 | (v69 >> 16) | HIDWORD(v69)) + 1;
    v71 = 0xFFFFFFFFLL;
    if ( v70 <= 0xFFFFFFFF )
      v71 = v70;
    v83 = v71;
    v38 = malloc(v71 << 6);
    if ( !v38 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v36 = *(unsigned int *)(a1 + 352);
    }
    v72 = *(_QWORD *)(a1 + 344);
    v73 = v72 + (v36 << 6);
    if ( v72 != v73 )
    {
      v74 = v73;
      v75 = v38;
      v81 = v7;
      v76 = *(char ***)(a1 + 344);
      do
      {
        if ( v75 )
        {
          v77 = *(_DWORD *)v76;
          *(_DWORD *)(v75 + 16) = 0;
          *(_DWORD *)(v75 + 20) = 1;
          *(_DWORD *)v75 = v77;
          *(_QWORD *)(v75 + 8) = v75 + 24;
          if ( *((_DWORD *)v76 + 4) )
            sub_3919AC0(v75 + 8, v76 + 1, v72, v21, (int)v24, v22);
          *(_DWORD *)(v75 + 40) = 0;
          *(_QWORD *)(v75 + 32) = v75 + 48;
          *(_DWORD *)(v75 + 44) = 4;
          v21 = *((unsigned int *)v76 + 10);
          if ( (_DWORD)v21 )
            sub_3919AC0(v75 + 32, v76 + 4, v72, v21, (int)v24, v22);
        }
        v76 += 8;
        v75 += 64;
      }
      while ( (char **)v74 != v76 );
      v8 = a1;
      v7 = v81;
      v73 = *(_QWORD *)(a1 + 344);
      if ( v73 + ((unsigned __int64)*(unsigned int *)(a1 + 352) << 6) != v73 )
      {
        v78 = v73 + ((unsigned __int64)*(unsigned int *)(a1 + 352) << 6);
        do
        {
          v78 -= 64LL;
          v79 = *(_QWORD *)(v78 + 32);
          if ( v79 != v78 + 48 )
            _libc_free(v79);
          v80 = *(_QWORD *)(v78 + 8);
          if ( v80 != v78 + 24 )
            _libc_free(v80);
        }
        while ( v73 != v78 );
        v8 = a1;
        v73 = *(_QWORD *)(a1 + 344);
      }
    }
    if ( v73 != v8 + 360 )
      _libc_free(v73);
    *(_QWORD *)(v8 + 344) = v38;
    LODWORD(v36) = *(_DWORD *)(v8 + 352);
    *(_DWORD *)(v8 + 356) = v83;
  }
  else
  {
    v38 = *(_QWORD *)(a1 + 344);
  }
  v39 = ((unsigned __int64)(unsigned int)v36 << 6) + v38;
  if ( v39 )
  {
    *(_DWORD *)v39 = v85;
    *(_QWORD *)(v39 + 8) = v39 + 24;
    *(_QWORD *)(v39 + 16) = 0x100000000LL;
    v40 = (unsigned int)v87;
    if ( (_DWORD)v87 )
      sub_39199E0(v39 + 8, (__int64)&v86, (unsigned int)v87, v21, (int)v24, v22);
    *(_QWORD *)(v39 + 32) = v39 + 48;
    *(_QWORD *)(v39 + 40) = 0x400000000LL;
    if ( (_DWORD)v90 )
      sub_39199E0(v39 + 32, (__int64)&v89, v40, v21, (int)v24, v22);
    LODWORD(v36) = *(_DWORD *)(v8 + 352);
  }
  v27 = *(_DWORD *)(v8 + 120);
  v29 = v8 + 96;
  *(_DWORD *)(v8 + 352) = v36 + 1;
  v28 = v23[16];
  if ( !v27 )
    goto LABEL_38;
LABEL_23:
  v30 = *(_QWORD *)(v8 + 104);
  v31 = (v27 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v32 = (__int64 *)(v30 + 16LL * v31);
  v33 = *v32;
  if ( v7 == *v32 )
    goto LABEL_24;
  v58 = 1;
  v59 = 0;
  while ( v33 != -8 )
  {
    if ( !v59 && v33 == -16 )
      v59 = v32;
    v31 = (v27 - 1) & (v58 + v31);
    v32 = (__int64 *)(v30 + 16LL * v31);
    v33 = *v32;
    if ( v7 == *v32 )
      goto LABEL_24;
    ++v58;
  }
  v60 = *(_DWORD *)(v8 + 112);
  if ( v59 )
    v32 = v59;
  ++*(_QWORD *)(v8 + 96);
  v45 = v60 + 1;
  if ( 4 * v45 >= 3 * v27 )
  {
LABEL_39:
    sub_391E830(v29, 2 * v27);
    v41 = *(_DWORD *)(v8 + 120);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(v8 + 104);
      v44 = v42 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v45 = *(_DWORD *)(v8 + 112) + 1;
      v32 = (__int64 *)(v43 + 16LL * v44);
      v46 = *v32;
      if ( v7 != *v32 )
      {
        v47 = 1;
        v48 = 0;
        while ( v46 != -8 )
        {
          if ( !v48 && v46 == -16 )
            v48 = v32;
          v44 = v42 & (v47 + v44);
          v32 = (__int64 *)(v43 + 16LL * v44);
          v46 = *v32;
          if ( v7 == *v32 )
            goto LABEL_60;
          ++v47;
        }
        if ( v48 )
          v32 = v48;
      }
      goto LABEL_60;
    }
    goto LABEL_110;
  }
  if ( v27 - *(_DWORD *)(v8 + 116) - v45 <= v27 >> 3 )
  {
    sub_391E830(v29, v27);
    v61 = *(_DWORD *)(v8 + 120);
    if ( v61 )
    {
      v62 = v61 - 1;
      v63 = *(_QWORD *)(v8 + 104);
      v64 = 1;
      v65 = v62 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v66 = 0;
      v45 = *(_DWORD *)(v8 + 112) + 1;
      v32 = (__int64 *)(v63 + 16LL * v65);
      v67 = *v32;
      if ( v7 != *v32 )
      {
        while ( v67 != -8 )
        {
          if ( !v66 && v67 == -16 )
            v66 = v32;
          v65 = v62 & (v64 + v65);
          v32 = (__int64 *)(v63 + 16LL * v65);
          v67 = *v32;
          if ( v7 == *v32 )
            goto LABEL_60;
          ++v64;
        }
        if ( v66 )
          v32 = v66;
      }
      goto LABEL_60;
    }
LABEL_110:
    ++*(_DWORD *)(v8 + 112);
    BUG();
  }
LABEL_60:
  *(_DWORD *)(v8 + 112) = v45;
  if ( *v32 != -8 )
    --*(_DWORD *)(v8 + 116);
  *v32 = v7;
  *((_DWORD *)v32 + 2) = 0;
LABEL_24:
  *((_DWORD *)v32 + 2) = v28;
  v34 = v23[16];
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  if ( v86 != &v88 )
    _libc_free((unsigned __int64)v86);
  return v34;
}
