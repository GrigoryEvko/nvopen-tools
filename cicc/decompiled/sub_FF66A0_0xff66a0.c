// Function: sub_FF66A0
// Address: 0xff66a0
//
__int64 __fastcall sub_FF66A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r15
  char v4; // al
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r8
  unsigned int v18; // eax
  _BYTE *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // r8d
  __int64 v25; // r13
  unsigned int v26; // r14d
  __int64 v27; // rbx
  __int64 v28; // rdx
  int v29; // r9d
  unsigned __int64 v30; // r8
  unsigned int *v31; // rax
  unsigned int v32; // esi
  unsigned int *v33; // rdi
  unsigned int *v34; // rdx
  unsigned int *v35; // rcx
  unsigned int *v36; // r10
  _BYTE *v37; // rdi
  unsigned int v38; // edx
  __int64 v39; // r8
  unsigned __int64 v40; // rax
  unsigned int *v41; // r9
  unsigned int *v42; // r11
  unsigned int v43; // r10d
  __int64 v44; // r15
  unsigned int *v45; // rsi
  unsigned int v46; // eax
  unsigned __int64 v47; // rcx
  __int64 v49; // r10
  __int64 v50; // rax
  __int64 v51; // r10
  __int64 v52; // [rsp+30h] [rbp-110h]
  int v53; // [rsp+30h] [rbp-110h]
  __int64 v54; // [rsp+38h] [rbp-108h]
  unsigned int v55; // [rsp+38h] [rbp-108h]
  __int64 v56; // [rsp+38h] [rbp-108h]
  unsigned __int64 v57; // [rsp+40h] [rbp-100h]
  __int64 v59; // [rsp+58h] [rbp-E8h]
  __int64 v60[2]; // [rsp+60h] [rbp-E0h] BYREF
  _BYTE *v61; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v62; // [rsp+78h] [rbp-C8h]
  _BYTE v63[16]; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int *v64; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+98h] [rbp-A8h]
  _BYTE v66[16]; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int *v67; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v68; // [rsp+B8h] [rbp-88h]
  _BYTE v69[16]; // [rsp+C0h] [rbp-80h] BYREF
  _DWORD v70[8]; // [rsp+D0h] [rbp-70h] BYREF
  _BYTE *v71; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v72; // [rsp+F8h] [rbp-48h]
  _BYTE v73[64]; // [rsp+100h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    goto LABEL_77;
  if ( !v2 )
    BUG();
  v3 = v2 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_77:
    BUG();
  v4 = *(_BYTE *)(v2 - 24);
  if ( (unsigned __int8)(v4 - 31) > 3u && v4 != 40 )
    return 0;
  v6 = sub_BC8A00(v3);
  if ( !v6 )
    return 0;
  v61 = v63;
  v64 = (unsigned int *)v66;
  v62 = 0x200000000LL;
  v65 = 0x200000000LL;
  v67 = (unsigned int *)v69;
  v68 = 0x200000000LL;
  sub_BC8BD0(v6, (__int64)&v61);
  if ( !(_DWORD)v62 )
    goto LABEL_62;
  v52 = (unsigned int)v62;
  v7 = 0;
  v54 = v3;
  v8 = a1;
  v57 = 0;
  do
  {
    while ( 1 )
    {
      v11 = v7;
      v57 += *(unsigned int *)&v61[4 * v7];
      sub_FEF2D0((__int64)v70, a2, *(_QWORD *)(v8 + 72), *(_QWORD *)(v8 + 80));
      v12 = sub_B46EC0(v54, v7);
      sub_FEF2D0((__int64)&v71, v12, *(_QWORD *)(v8 + 72), *(_QWORD *)(v8 + 80));
      v60[0] = (__int64)v70;
      v60[1] = (__int64)&v71;
      v59 = sub_FEF7A0(v8, v60);
      if ( !BYTE4(v59) || (_DWORD)v59 )
        break;
      v15 = (unsigned int)v65;
      v16 = (unsigned int)v65 + 1LL;
      if ( v16 > HIDWORD(v65) )
      {
        sub_C8D5F0((__int64)&v64, v66, v16, 4u, v13, v14);
        v15 = (unsigned int)v65;
      }
      ++v7;
      v64[v15] = v11;
      LODWORD(v65) = v65 + 1;
      if ( v52 == v7 )
        goto LABEL_17;
    }
    v9 = (unsigned int)v68;
    v10 = (unsigned int)v68 + 1LL;
    if ( v10 > HIDWORD(v68) )
    {
      sub_C8D5F0((__int64)&v67, v69, v10, 4u, v13, v14);
      v9 = (unsigned int)v68;
    }
    ++v7;
    v67[v9] = v11;
    LODWORD(v68) = v68 + 1;
  }
  while ( v52 != v7 );
LABEL_17:
  v17 = v57;
  a1 = v8;
  v3 = v54;
  if ( v57 > 0xFFFFFFFF )
  {
    v18 = sub_B46E30(v54);
    if ( !v18 )
    {
LABEL_66:
      LODWORD(v17) = v18;
      goto LABEL_23;
    }
    v19 = v61;
    v20 = 0;
    v17 = 0;
    v21 = 4LL * v18;
    do
    {
      *(_DWORD *)&v19[v20] = *(unsigned int *)&v19[v20] / (v57 / 0xFFFFFFFF + 1);
      v19 = v61;
      v22 = *(unsigned int *)&v61[v20];
      v20 += 4;
      v17 += v22;
    }
    while ( v21 != v20 );
  }
  if ( !v17 || !(_DWORD)v68 )
  {
LABEL_62:
    v18 = sub_B46E30(v3);
    if ( v18 )
    {
      v49 = v18;
      v50 = 0;
      v51 = 4 * v49;
      do
      {
        *(_DWORD *)&v61[v50] = 1;
        v50 += 4;
      }
      while ( v51 != v50 );
      v18 = sub_B46E30(v3);
    }
    goto LABEL_66;
  }
LABEL_23:
  v55 = v17;
  v71 = v73;
  v72 = 0x200000000LL;
  v23 = sub_B46E30(v3);
  v24 = v55;
  if ( v23 )
  {
    v56 = a1;
    v25 = 0;
    v26 = v24;
    v27 = 4LL * v23;
    do
    {
      sub_F02DB0(v70, *(_DWORD *)&v61[v25], v26);
      v28 = (unsigned int)v72;
      v29 = v70[0];
      v30 = (unsigned int)v72 + 1LL;
      if ( v30 > HIDWORD(v72) )
      {
        v53 = v70[0];
        sub_C8D5F0((__int64)&v71, v73, (unsigned int)v72 + 1LL, 4u, v30, v70[0]);
        v28 = (unsigned int)v72;
        v29 = v53;
      }
      v25 += 4;
      *(_DWORD *)&v71[4 * v28] = v29;
      LODWORD(v72) = v72 + 1;
    }
    while ( v27 != v25 );
    a1 = v56;
  }
  if ( !(_DWORD)v65 || !(_DWORD)v68 )
    goto LABEL_53;
  v31 = v64;
  v32 = dword_4F8E738;
  v33 = &v64[(unsigned int)v65];
  do
  {
    v34 = (unsigned int *)&v71[4 * *v31];
    if ( v32 < *v34 )
      *v34 = v32;
    ++v31;
  }
  while ( v33 != v31 );
  v35 = v64;
  v36 = &v64[(unsigned int)v65];
  if ( v36 == v64 )
  {
    v44 = 0x80000000LL;
  }
  else
  {
    v37 = v71;
    v38 = 0;
    do
    {
      v39 = *(unsigned int *)&v71[4 * *v35];
      v40 = v39 + v38;
      v38 += v39;
      if ( v40 > 0x80000000 )
        v38 = 0x80000000;
      ++v35;
    }
    while ( v36 != v35 );
    if ( v38 > 0x80000000 )
    {
      v41 = v67;
      v42 = &v67[(unsigned int)v68];
      v43 = v68;
      if ( v42 != v67 )
      {
        v44 = 0;
        goto LABEL_43;
      }
      goto LABEL_53;
    }
    v44 = 0x80000000 - v38;
  }
  v41 = v67;
  v42 = &v67[(unsigned int)v68];
  v43 = v68;
  if ( v42 != v67 )
  {
    v37 = v71;
LABEL_43:
    v45 = v41;
    v46 = 0;
    while ( 1 )
    {
      v47 = *(unsigned int *)&v37[4 * *v45] + (unsigned __int64)v46;
      v46 += *(_DWORD *)&v37[4 * *v45];
      if ( v47 > 0x80000000 )
        v46 = 0x80000000;
      if ( v42 == v45 + 1 )
        break;
      ++v45;
    }
    if ( v46 != (_DWORD)v44 )
    {
      if ( v46 )
      {
        while ( 1 )
        {
          *(_DWORD *)&v37[4 * *v41] = (v44 * (unsigned __int64)*(unsigned int *)&v37[4 * *v41] % v46 > ((unsigned __int64)v46 - 1) >> 1)
                                    + (unsigned int)(v44 * (unsigned __int64)*(unsigned int *)&v37[4 * *v41] / v46);
          if ( v45 == v41 )
            break;
          v37 = v71;
          ++v41;
        }
      }
      else
      {
        while ( 1 )
        {
          *(_DWORD *)&v37[4 * *v41] = (unsigned int)v44 / v43;
          if ( v41 == v45 )
            break;
          v37 = v71;
          ++v41;
        }
      }
    }
  }
LABEL_53:
  sub_FF6650(a1, a2, (__int64)&v71);
  if ( v71 != v73 )
    _libc_free(v71, a2);
  if ( v67 != (unsigned int *)v69 )
    _libc_free(v67, a2);
  if ( v64 != (unsigned int *)v66 )
    _libc_free(v64, a2);
  if ( v61 != v63 )
    _libc_free(v61, a2);
  return 1;
}
