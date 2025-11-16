// Function: sub_D63080
// Address: 0xd63080
//
__int64 __fastcall sub_D63080(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // r14
  unsigned __int8 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  unsigned int v10; // ecx
  unsigned __int8 **v11; // rsi
  unsigned __int8 *v12; // rdi
  char *v13; // r12
  __int64 v14; // rsi
  char *v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rdi
  int v19; // esi
  __int64 v20; // rax
  __int16 v21; // ax
  unsigned __int8 **v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int8 *v26; // rsi
  unsigned __int8 **v27; // rax
  __int64 v28; // rbx
  int v29; // eax
  int v30; // esi
  int v31; // r10d
  _QWORD *v32; // r9
  __int64 v33; // r8
  unsigned int v34; // eax
  _QWORD *v35; // rcx
  unsigned __int8 *v36; // rdx
  unsigned __int64 *v37; // r12
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // r8
  unsigned __int64 v44; // rsi
  _QWORD *v45; // rax
  int v46; // ecx
  _QWORD *v47; // rdx
  char v48; // dl
  unsigned __int8 v49; // al
  unsigned __int64 v50; // rdx
  bool v51; // al
  unsigned __int64 v52; // rax
  _QWORD *v53; // rbx
  __int64 v54; // rax
  bool v55; // zf
  unsigned __int64 v56; // rax
  int v57; // r10d
  unsigned __int64 v58; // rsi
  unsigned __int64 *v59; // [rsp+0h] [rbp-2A0h]
  __int64 v60; // [rsp+0h] [rbp-2A0h]
  unsigned __int8 *v61; // [rsp+8h] [rbp-298h] BYREF
  __int64 v62; // [rsp+10h] [rbp-290h] BYREF
  __int64 v63; // [rsp+18h] [rbp-288h]
  __int64 v64; // [rsp+20h] [rbp-280h]
  __int64 v65; // [rsp+30h] [rbp-270h] BYREF
  __int64 v66; // [rsp+38h] [rbp-268h]
  unsigned __int64 v67; // [rsp+40h] [rbp-260h]
  __int64 v68; // [rsp+50h] [rbp-250h] BYREF
  unsigned int v69; // [rsp+58h] [rbp-248h]
  __int64 v70; // [rsp+60h] [rbp-240h] BYREF
  unsigned int v71; // [rsp+68h] [rbp-238h]
  __int64 v72; // [rsp+70h] [rbp-230h] BYREF
  __int64 v73; // [rsp+78h] [rbp-228h]
  __int64 v74; // [rsp+80h] [rbp-220h]
  unsigned __int64 v75[2]; // [rsp+88h] [rbp-218h] BYREF
  unsigned __int64 v76; // [rsp+98h] [rbp-208h]
  __int64 v77; // [rsp+A0h] [rbp-200h] BYREF
  _QWORD v78[2]; // [rsp+A8h] [rbp-1F8h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-1E8h]
  __int64 v80; // [rsp+C0h] [rbp-1E0h]
  __int16 v81; // [rsp+C8h] [rbp-1D8h]
  __int64 v82[2]; // [rsp+D0h] [rbp-1D0h] BYREF
  _QWORD v83[5]; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v84; // [rsp+108h] [rbp-198h]
  unsigned int v85; // [rsp+110h] [rbp-190h]
  char v86; // [rsp+120h] [rbp-180h]
  char *v87; // [rsp+128h] [rbp-178h] BYREF
  unsigned int v88; // [rsp+130h] [rbp-170h]
  char v89; // [rsp+268h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_QWORD *)(a1 + 8);
  v61 = a2;
  v5 = *(_QWORD *)(a1 + 352);
  LOBYTE(v5) = 1;
  sub_D5D740(v83, *(_QWORD *)a1, v4, v3, v5, *(_QWORD *)(a1 + 360));
  sub_D62AB0((__int64)&v68, (__int64)v83, (__int64)a2);
  if ( v69 > 1 && v71 > 1 )
  {
    sub_ACCFD0(*(__int64 **)(a1 + 16), (__int64)&v70);
    v6 = sub_ACCFD0(*(__int64 **)(a1 + 16), (__int64)&v68);
    goto LABEL_8;
  }
  v7 = sub_BD3990(v61, (__int64)v83);
  v8 = *(unsigned int *)(a1 + 248);
  v9 = *(_QWORD *)(a1 + 232);
  v61 = v7;
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v11 = (unsigned __int8 **)(v9 + 56LL * v10);
    v12 = *v11;
    if ( v7 == *v11 )
    {
LABEL_6:
      if ( v11 != (unsigned __int8 **)(v9 + 56 * v8) )
      {
        sub_D5EB80(&v77, (__int64)(v11 + 1));
        v6 = v77;
        goto LABEL_8;
      }
    }
    else
    {
      v19 = 1;
      while ( v12 != (unsigned __int8 *)-4096LL )
      {
        v57 = v19 + 1;
        v10 = (v8 - 1) & (v19 + v10);
        v11 = (unsigned __int8 **)(v9 + 56LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
          goto LABEL_6;
        v19 = v57;
      }
    }
  }
  v20 = *(_QWORD *)(a1 + 72);
  v78[0] = 0;
  v77 = a1 + 24;
  v79 = v20;
  v78[1] = 0;
  if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
    sub_BD73F0((__int64)v78);
  v21 = *(_WORD *)(a1 + 88);
  v80 = *(_QWORD *)(a1 + 80);
  v81 = v21;
  sub_B33910(v82, (__int64 *)(a1 + 24));
  v26 = v61;
  if ( *v61 > 0x1Cu )
  {
    sub_D5F1F0(a1 + 24, (__int64)v61);
    v26 = v61;
  }
  if ( !*(_BYTE *)(a1 + 284) )
    goto LABEL_97;
  v27 = *(unsigned __int8 ***)(a1 + 264);
  v23 = *(unsigned int *)(a1 + 276);
  v22 = &v27[v23];
  if ( v27 != v22 )
  {
    while ( *v27 != v26 )
    {
      if ( v22 == ++v27 )
        goto LABEL_96;
    }
    goto LABEL_46;
  }
LABEL_96:
  if ( (unsigned int)v23 < *(_DWORD *)(a1 + 272) )
  {
    *(_DWORD *)(a1 + 276) = v23 + 1;
    *v22 = v26;
    ++*(_QWORD *)(a1 + 256);
  }
  else
  {
LABEL_97:
    sub_C8CC70(a1 + 256, (__int64)v26, (__int64)v22, v23, v24, v25);
    if ( !v48 )
      goto LABEL_46;
  }
  v49 = *v61;
  if ( *v61 <= 0x1Cu )
  {
    if ( v49 == 5 && *((_WORD *)v61 + 1) == 34 )
      goto LABEL_101;
LABEL_46:
    v65 = 6;
    v6 = 0;
    v28 = 0;
    v66 = 0;
    v67 = 0;
    v62 = 6;
    v63 = 0;
    v64 = 0;
LABEL_47:
    v72 = 6;
    v73 = 0;
    v74 = v28;
    goto LABEL_48;
  }
  if ( v49 != 63 )
  {
    v28 = sub_D65D10(a1);
    v6 = v28;
    goto LABEL_102;
  }
LABEL_101:
  v28 = sub_D64C90(a1);
  v6 = v28;
LABEL_102:
  v67 = v50;
  v65 = 6;
  v66 = 0;
  v51 = v28 != -8192 && v28 != 0 && v28 != -4096;
  if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
  {
    sub_BD73F0((__int64)&v65);
    v51 = v28 != -8192 && v28 != 0 && v28 != -4096;
  }
  v62 = 6;
  v63 = 0;
  v64 = v28;
  if ( !v51 )
    goto LABEL_47;
  sub_BD73F0((__int64)&v62);
  v72 = 6;
  v73 = 0;
  v74 = v64;
  if ( v64 != -4096 && v64 != 0 && v64 != -8192 )
    sub_BD6050((unsigned __int64 *)&v72, v62 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_48:
  v75[0] = 6;
  v75[1] = 0;
  v76 = v67;
  if ( v67 != -4096 && v67 != 0 && v67 != -8192 )
    sub_BD6050(v75, v65 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v64 != 0 && v64 != -4096 && v64 != -8192 )
    sub_BD60C0(&v62);
  if ( v67 != 0 && v67 != -4096 && v67 != -8192 )
    sub_BD60C0(&v65);
  v29 = *(_DWORD *)(a1 + 248);
  if ( v29 )
  {
    v30 = v29 - 1;
    v31 = 1;
    v32 = 0;
    v33 = *(_QWORD *)(a1 + 232);
    v34 = (v29 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
    v35 = (_QWORD *)(v33 + 56LL * v34);
    v36 = (unsigned __int8 *)*v35;
    if ( (unsigned __int8 *)*v35 == v61 )
    {
LABEL_59:
      v37 = v35 + 1;
      v59 = v35 + 4;
      goto LABEL_60;
    }
    while ( v36 != (unsigned __int8 *)-4096LL )
    {
      if ( v36 == (unsigned __int8 *)-8192LL && !v32 )
        v32 = v35;
      v34 = v30 & (v31 + v34);
      v35 = (_QWORD *)(v33 + 56LL * v34);
      v36 = (unsigned __int8 *)*v35;
      if ( v61 == (unsigned __int8 *)*v35 )
        goto LABEL_59;
      ++v31;
    }
    if ( !v32 )
      v32 = v35;
  }
  else
  {
    v32 = 0;
  }
  v53 = sub_D5FA60(a1 + 224, &v61, v32);
  v37 = v53 + 1;
  *v53 = v61;
  v64 = 0;
  v65 = 6;
  v66 = 0;
  v67 = 0;
  v62 = 6;
  v63 = 0;
  v53[1] = 6;
  v53[2] = 0;
  v54 = v64;
  v55 = v64 == -4096;
  v53[3] = v64;
  if ( v54 != 0 && !v55 && v54 != -8192 )
    sub_BD6050(v37, 0);
  v53[4] = 6;
  v53[5] = 0;
  v56 = v67;
  v59 = v53 + 4;
  v55 = v67 == -4096;
  v53[6] = v67;
  if ( v56 != 0 && !v55 && v56 != -8192 )
    sub_BD6050(v53 + 4, v65 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v64 != -4096 && v64 != 0 && v64 != -8192 )
    sub_BD60C0(&v62);
  if ( v67 != 0 && v67 != -4096 && v67 != -8192 )
    sub_BD60C0(&v65);
LABEL_60:
  v38 = v37[2];
  v39 = v74;
  if ( v38 != v74 )
  {
    if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
    {
      sub_BD60C0(v37);
      v39 = v74;
    }
    v37[2] = v39;
    if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
      sub_BD6050(v37, v72 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v40 = v37[5];
  v41 = v76;
  if ( v40 != v76 )
  {
    if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
    {
      sub_BD60C0(v59);
      v41 = v76;
    }
    v37[5] = v41;
    if ( v41 != 0 && v41 != -4096 && v41 != -8192 )
      sub_BD6050(v59, v75[0] & 0xFFFFFFFFFFFFFFF8LL);
    v41 = v76;
  }
  if ( v41 != 0 && v41 != -4096 && v41 != -8192 )
    sub_BD60C0(v75);
  if ( v74 != 0 && v74 != -4096 && v74 != -8192 )
    sub_BD60C0(&v72);
  v42 = v77;
  if ( v79 )
  {
    sub_A88F30(v77, v79, v80, v81);
    v42 = v77;
  }
  else
  {
    *(_QWORD *)(v77 + 48) = 0;
    *(_QWORD *)(v42 + 56) = 0;
    *(_WORD *)(v42 + 64) = 0;
  }
  v72 = v82[0];
  if ( !v82[0] || (sub_B96E90((__int64)&v72, v82[0], 1), (v43 = v72) == 0) )
  {
    sub_93FB40(v42, 0);
    v43 = v72;
    goto LABEL_113;
  }
  v44 = *(unsigned int *)(v42 + 8);
  v45 = *(_QWORD **)v42;
  v46 = *(_DWORD *)(v42 + 8);
  v47 = (_QWORD *)(*(_QWORD *)v42 + 16 * v44);
  if ( *(_QWORD **)v42 == v47 )
  {
LABEL_109:
    v52 = *(unsigned int *)(v42 + 12);
    if ( v44 >= v52 )
    {
      v58 = v44 + 1;
      if ( v52 < v58 )
      {
        v60 = v72;
        sub_C8D5F0(v42, (const void *)(v42 + 16), v58, 0x10u, v72, v42 + 16);
        v43 = v60;
        v47 = (_QWORD *)(*(_QWORD *)v42 + 16LL * *(unsigned int *)(v42 + 8));
      }
      *v47 = 0;
      v47[1] = v43;
      ++*(_DWORD *)(v42 + 8);
      v43 = v72;
    }
    else
    {
      if ( v47 )
      {
        *(_DWORD *)v47 = 0;
        v47[1] = v43;
        v46 = *(_DWORD *)(v42 + 8);
        v43 = v72;
      }
      *(_DWORD *)(v42 + 8) = v46 + 1;
    }
LABEL_113:
    if ( !v43 )
      goto LABEL_91;
    goto LABEL_90;
  }
  while ( *(_DWORD *)v45 )
  {
    v45 += 2;
    if ( v47 == v45 )
      goto LABEL_109;
  }
  v45[1] = v72;
LABEL_90:
  sub_B91220((__int64)&v72, v43);
LABEL_91:
  if ( v82[0] )
    sub_B91220((__int64)v82, v82[0]);
  if ( v79 != -4096 && v79 != 0 && v79 != -8192 )
    sub_BD60C0(v78);
LABEL_8:
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  if ( (v86 & 1) != 0 )
  {
    v15 = &v89;
    v13 = (char *)&v87;
  }
  else
  {
    v13 = v87;
    v14 = 40LL * v88;
    if ( !v88 )
      goto LABEL_32;
    v15 = &v87[v14];
    if ( &v87[v14] == v87 )
      goto LABEL_32;
  }
  do
  {
    if ( *(_QWORD *)v13 != -8192 && *(_QWORD *)v13 != -4096 )
    {
      if ( *((_DWORD *)v13 + 8) > 0x40u )
      {
        v16 = *((_QWORD *)v13 + 3);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      if ( *((_DWORD *)v13 + 4) > 0x40u )
      {
        v17 = *((_QWORD *)v13 + 1);
        if ( v17 )
          j_j___libc_free_0_0(v17);
      }
    }
    v13 += 40;
  }
  while ( v15 != v13 );
  if ( (v86 & 1) == 0 )
  {
    v13 = v87;
    v14 = 40LL * v88;
LABEL_32:
    sub_C7D6A0((__int64)v13, v14, 8);
  }
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  return v6;
}
