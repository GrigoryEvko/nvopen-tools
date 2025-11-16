// Function: sub_2CC3FF0
// Address: 0x2cc3ff0
//
__int64 __fastcall sub_2CC3FF0(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // rsi
  __int64 v4; // rdi
  unsigned int v5; // eax
  __int64 v6; // rdx
  unsigned int v7; // r15d
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 i; // r12
  __int64 v32; // rax
  unsigned int v33; // esi
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // rcx
  __int64 v39; // rbx
  __int64 j; // r12
  __int64 v41; // rax
  unsigned int v42; // esi
  unsigned __int64 v43; // rbx
  __int64 v44; // rsi
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // r12
  _QWORD *v48; // rax
  __int64 v49; // rbx
  unsigned __int64 v50; // r14
  _BYTE *v51; // r12
  __int64 v52; // rdx
  unsigned int v53; // esi
  __int64 *v54; // r12
  __int64 v55; // rbx
  __int64 k; // r12
  __int64 v57; // rax
  unsigned int v58; // esi
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // [rsp+0h] [rbp-1B0h]
  __int64 v68; // [rsp+20h] [rbp-190h]
  __int64 v69; // [rsp+30h] [rbp-180h] BYREF
  unsigned __int8 *v70; // [rsp+38h] [rbp-178h] BYREF
  __int64 v71; // [rsp+40h] [rbp-170h] BYREF
  __int64 v72; // [rsp+48h] [rbp-168h] BYREF
  __int64 v73; // [rsp+50h] [rbp-160h] BYREF
  __int64 v74; // [rsp+58h] [rbp-158h] BYREF
  __int64 v75; // [rsp+60h] [rbp-150h] BYREF
  __int64 v76; // [rsp+68h] [rbp-148h] BYREF
  __int64 v77; // [rsp+70h] [rbp-140h] BYREF
  __int64 v78; // [rsp+78h] [rbp-138h] BYREF
  __int64 v79; // [rsp+80h] [rbp-130h] BYREF
  __int64 v80; // [rsp+88h] [rbp-128h] BYREF
  __int64 v81; // [rsp+90h] [rbp-120h] BYREF
  __int64 v82; // [rsp+98h] [rbp-118h] BYREF
  __int64 v83; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v84; // [rsp+A8h] [rbp-108h] BYREF
  __int64 v85; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v86; // [rsp+B8h] [rbp-F8h] BYREF
  __int64 v87[4]; // [rsp+C0h] [rbp-F0h] BYREF
  __int16 v88; // [rsp+E0h] [rbp-D0h]
  _BYTE *v89; // [rsp+F0h] [rbp-C0h]
  __int64 v90; // [rsp+F8h] [rbp-B8h]
  _BYTE v91[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v92; // [rsp+120h] [rbp-90h]
  __int64 v93; // [rsp+128h] [rbp-88h]
  __int64 v94; // [rsp+130h] [rbp-80h]
  __int64 v95; // [rsp+138h] [rbp-78h]
  void **v96; // [rsp+140h] [rbp-70h]
  void **v97; // [rsp+148h] [rbp-68h]
  __int64 v98; // [rsp+150h] [rbp-60h]
  int v99; // [rsp+158h] [rbp-58h]
  __int16 v100; // [rsp+15Ch] [rbp-54h]
  char v101; // [rsp+15Eh] [rbp-52h]
  __int64 v102; // [rsp+160h] [rbp-50h]
  __int64 v103; // [rsp+168h] [rbp-48h]
  void *v104; // [rsp+170h] [rbp-40h] BYREF
  void *v105; // [rsp+178h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 160);
  v3 = *(_QWORD *)(a1 + 24);
  v69 = 0;
  v70 = 0;
  if ( v2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(v2 + 44) + 1);
    v5 = *(_DWORD *)(v2 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  if ( v5 >= *(_DWORD *)(v3 + 32) )
    BUG();
  v6 = *(_QWORD *)(a1 + 152);
  if ( v6 != **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v3 + 24) + 8 * v4) + 8LL) )
    return 0;
  v9 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v6 + 48 )
    goto LABEL_131;
  if ( !v9 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_131:
    BUG();
  if ( *(_BYTE *)(v9 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v10 = *(_QWORD *)(v9 - 120);
  if ( *(_BYTE *)v10 != 82 )
    return 0;
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(_QWORD *)a1;
  v13 = *(_WORD *)(v10 + 2) & 0x3F;
  if ( (*(_WORD *)(v10 + 2) & 0x3F) == 0x21 )
  {
    if ( v2 != *(_QWORD *)(v9 - 88) )
      return 0;
  }
  else if ( (*(_WORD *)(v10 + 2) & 0x3F) != 0x20 || v2 != *(_QWORD *)(v9 - 56) )
  {
    return 0;
  }
  if ( v11 == *(_QWORD *)(v10 - 64) && (unsigned __int8)sub_D48480(*(_QWORD *)a1, *(_QWORD *)(v10 - 32), v13, v2) )
  {
    v67 = *(_QWORD *)(v10 - 32);
  }
  else
  {
    if ( v11 != *(_QWORD *)(v10 - 32) || !(unsigned __int8)sub_D48480(v12, *(_QWORD *)(v10 - 64), v13, v2) )
      return 0;
    v67 = *(_QWORD *)(v10 - 64);
  }
  if ( !(unsigned __int8)sub_2CBFC80(
                           *(_QWORD *)a1,
                           *(_QWORD *)(a1 + 176),
                           *(_QWORD *)(a1 + 144),
                           *(_QWORD *)(a1 + 152),
                           *(_QWORD *)(a1 + 160),
                           *(_QWORD *)(a1 + 168),
                           *(_QWORD *)(a1 + 184),
                           *(_QWORD *)(a1 + 56),
                           *(_QWORD *)(a1 + 64),
                           &v69,
                           &v70) )
    return 0;
  v7 = sub_2CBF770(
         *(_QWORD *)a1,
         *(_QWORD *)(a1 + 176),
         *(_QWORD *)(a1 + 152),
         *(__int64 **)(a1 + 160),
         *(_QWORD *)(a1 + 168),
         *(_QWORD *)(a1 + 184));
  if ( !(_BYTE)v7 )
    return 0;
  v14 = *(_QWORD *)(a1 + 40);
  if ( !v14 )
    return 0;
  if ( *(_QWORD *)(a1 + 152) != *(_QWORD *)(v14 + 40) )
    return 0;
  v15 = *(_QWORD *)(a1 + 184);
  v16 = *(_QWORD *)(a1 + 168);
  v17 = *(_QWORD *)(a1 + 160);
  v68 = sub_2CBF180(v14, v17, v16, v15);
  if ( !v68 )
    return 0;
  sub_2CC1B10(
    *(_QWORD *)a1,
    &v84,
    &v85,
    1,
    *(__int64 **)(a1 + 8),
    *(_QWORD *)(a1 + 200),
    *(_QWORD *)(a1 + 16),
    v68,
    &v83,
    *(_QWORD *)(a1 + 176),
    *(_QWORD *)(a1 + 144),
    v18,
    v17,
    v16,
    v15,
    *(_QWORD *)(a1 + 192),
    &v71,
    &v72,
    (__int64)&v73,
    &v74,
    &v75,
    &v76,
    &v77,
    &v78,
    (__int64)&v79,
    &v80,
    &v81,
    &v82);
  v21 = *(unsigned int *)(a1 + 216);
  v22 = v85;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 220) )
  {
    sub_C8D5F0(a1 + 208, (const void *)(a1 + 224), v21 + 1, 8u, v19, v20);
    v21 = *(unsigned int *)(a1 + 216);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 208) + 8 * v21) = v22;
  v23 = *(_QWORD *)(a1 + 152);
  ++*(_DWORD *)(a1 + 216);
  v24 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 == v23 + 48 )
    goto LABEL_135;
  if ( !v24 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 > 0xA )
LABEL_135:
    BUG();
  v25 = *(_QWORD *)(v24 - 56);
  v26 = *(_QWORD *)(v24 - 88);
  if ( *(_QWORD *)(a1 + 160) == v25 )
  {
    if ( v26 )
    {
      if ( v25 )
      {
        v63 = *(_QWORD *)(v24 - 48);
        **(_QWORD **)(v24 - 40) = v63;
        if ( v63 )
          *(_QWORD *)(v63 + 16) = *(_QWORD *)(v24 - 40);
      }
      *(_QWORD *)(v24 - 56) = v26;
      v64 = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v24 - 48) = v64;
      if ( v64 )
        *(_QWORD *)(v64 + 16) = v24 - 48;
      *(_QWORD *)(v24 - 40) = v26 + 16;
      *(_QWORD *)(v26 + 16) = v24 - 56;
    }
    else if ( v25 )
    {
      v65 = *(_QWORD *)(v24 - 48);
      **(_QWORD **)(v24 - 40) = v65;
      if ( v65 )
        *(_QWORD *)(v65 + 16) = *(_QWORD *)(v24 - 40);
      *(_QWORD *)(v24 - 56) = 0;
    }
  }
  else
  {
    if ( v26 )
    {
      v27 = *(_QWORD *)(v24 - 80);
      **(_QWORD **)(v24 - 72) = v27;
      if ( v27 )
        *(_QWORD *)(v27 + 16) = *(_QWORD *)(v24 - 72);
    }
    *(_QWORD *)(v24 - 88) = v25;
    if ( v25 )
    {
      v28 = *(_QWORD *)(v25 + 16);
      *(_QWORD *)(v24 - 80) = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 16) = v24 - 80;
      *(_QWORD *)(v24 - 72) = v25 + 16;
      *(_QWORD *)(v25 + 16) = v24 - 88;
    }
  }
  v29 = *(_QWORD *)(a1 + 160);
  v30 = *(_QWORD *)(v29 + 56);
  for ( i = v29 + 48; i != v30; v30 = *(_QWORD *)(v30 + 8) )
  {
    if ( !v30 )
      BUG();
    if ( *(_BYTE *)(v30 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v30 - 20) & 0x7FFFFFF) != 0 )
    {
      v32 = 0;
      while ( 1 )
      {
        v33 = v32;
        if ( *(_QWORD *)(a1 + 152) == *(_QWORD *)(*(_QWORD *)(v30 - 32) + 32LL * *(unsigned int *)(v30 + 48) + 8 * v32) )
          break;
        if ( (*(_DWORD *)(v30 - 20) & 0x7FFFFFF) == (_DWORD)++v32 )
          goto LABEL_93;
      }
    }
    else
    {
LABEL_93:
      v33 = -1;
    }
    sub_B48BF0(v30 - 24, v33, 1);
  }
  v34 = *(_QWORD *)(v79 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v34 == v79 + 48 )
    goto LABEL_141;
  if ( !v34 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v34 - 24) - 30 > 0xA )
LABEL_141:
    BUG();
  v35 = *(_QWORD *)(v34 - 56);
  v36 = *(_QWORD *)(v34 - 88);
  if ( v80 == v35 )
  {
    if ( v36 )
    {
      if ( v35 )
      {
        v59 = *(_QWORD *)(v34 - 48);
        **(_QWORD **)(v34 - 40) = v59;
        if ( v59 )
          *(_QWORD *)(v59 + 16) = *(_QWORD *)(v34 - 40);
      }
      *(_QWORD *)(v34 - 56) = v36;
      v60 = *(_QWORD *)(v36 + 16);
      *(_QWORD *)(v34 - 48) = v60;
      if ( v60 )
        *(_QWORD *)(v60 + 16) = v34 - 48;
      *(_QWORD *)(v34 - 40) = v36 + 16;
      *(_QWORD *)(v36 + 16) = v34 - 56;
    }
    else if ( v35 )
    {
      v66 = *(_QWORD *)(v34 - 48);
      **(_QWORD **)(v34 - 40) = v66;
      if ( v66 )
        *(_QWORD *)(v66 + 16) = *(_QWORD *)(v34 - 40);
      *(_QWORD *)(v34 - 56) = 0;
    }
  }
  else
  {
    if ( v36 )
    {
      v37 = *(_QWORD *)(v34 - 80);
      **(_QWORD **)(v34 - 72) = v37;
      if ( v37 )
        *(_QWORD *)(v37 + 16) = *(_QWORD *)(v34 - 72);
    }
    *(_QWORD *)(v34 - 88) = v35;
    if ( v35 )
    {
      v38 = *(_QWORD *)(v35 + 16);
      *(_QWORD *)(v34 - 80) = v38;
      if ( v38 )
        *(_QWORD *)(v38 + 16) = v34 - 80;
      *(_QWORD *)(v34 - 72) = v35 + 16;
      *(_QWORD *)(v35 + 16) = v34 - 88;
    }
  }
  v39 = *(_QWORD *)(v80 + 56);
  for ( j = v80 + 48; j != v39; v39 = *(_QWORD *)(v39 + 8) )
  {
    if ( !v39 )
      BUG();
    if ( *(_BYTE *)(v39 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v39 - 20) & 0x7FFFFFF) != 0 )
    {
      v41 = 0;
      while ( 1 )
      {
        v42 = v41;
        if ( v79 == *(_QWORD *)(*(_QWORD *)(v39 - 32) + 32LL * *(unsigned int *)(v39 + 48) + 8 * v41) )
          break;
        if ( (*(_DWORD *)(v39 - 20) & 0x7FFFFFF) == (_DWORD)++v41 )
          goto LABEL_92;
      }
    }
    else
    {
LABEL_92:
      v42 = -1;
    }
    sub_B48BF0(v39 - 24, v42, 1);
  }
  v43 = *(_QWORD *)(v74 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v43 == v74 + 48 )
    goto LABEL_137;
  if ( !v43 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 > 0xA )
LABEL_137:
    BUG();
  v44 = *(_QWORD *)(v43 + 24);
  v86 = v44;
  if ( v44 )
    sub_B96E90((__int64)&v86, v44, 1);
  sub_B43D60((_QWORD *)(v43 - 24));
  v45 = v74;
  v46 = sub_AA48A0(v74);
  v101 = 7;
  v95 = v46;
  v47 = v75;
  v96 = &v104;
  v97 = &v105;
  v89 = v91;
  v104 = &unk_49DA100;
  v90 = 0x200000000LL;
  v92 = v45;
  v88 = 257;
  v93 = v45 + 48;
  LOWORD(v94) = 0;
  v98 = 0;
  v99 = 0;
  v100 = 512;
  v102 = 0;
  v103 = 0;
  v105 = &unk_49DA0B0;
  v48 = sub_BD2C40(72, 1u);
  v49 = (__int64)v48;
  if ( v48 )
    sub_B4C8F0((__int64)v48, v47, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v97 + 2))(v97, v49, v87, v93, v94);
  v50 = (unsigned __int64)v89;
  v51 = &v89[16 * (unsigned int)v90];
  if ( v89 != v51 )
  {
    do
    {
      v52 = *(_QWORD *)(v50 + 8);
      v53 = *(_DWORD *)v50;
      v50 += 16LL;
      sub_B99FD0(v49, v53, v52);
    }
    while ( v51 != (_BYTE *)v50 );
  }
  v54 = (__int64 *)(v49 + 48);
  v87[0] = v86;
  if ( v86 )
  {
    sub_B96E90((__int64)v87, v86, 1);
    if ( v54 == v87 )
    {
      if ( v87[0] )
        sub_B91220((__int64)v87, v87[0]);
      goto LABEL_78;
    }
    v61 = *(_QWORD *)(v49 + 48);
    if ( !v61 )
    {
LABEL_104:
      v62 = (unsigned __int8 *)v87[0];
      *(_QWORD *)(v49 + 48) = v87[0];
      if ( v62 )
        sub_B976B0((__int64)v87, v62, v49 + 48);
      goto LABEL_78;
    }
LABEL_103:
    sub_B91220(v49 + 48, v61);
    goto LABEL_104;
  }
  if ( v54 != v87 )
  {
    v61 = *(_QWORD *)(v49 + 48);
    if ( v61 )
      goto LABEL_103;
  }
LABEL_78:
  v55 = *(_QWORD *)(v73 + 56);
  for ( k = v73 + 48; k != v55; v55 = *(_QWORD *)(v55 + 8) )
  {
    if ( !v55 )
      BUG();
    if ( *(_BYTE *)(v55 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v55 - 20) & 0x7FFFFFF) != 0 )
    {
      v57 = 0;
      while ( 1 )
      {
        v58 = v57;
        if ( v74 == *(_QWORD *)(*(_QWORD *)(v55 - 32) + 32LL * *(unsigned int *)(v55 + 48) + 8 * v57) )
          break;
        if ( (*(_DWORD *)(v55 - 20) & 0x7FFFFFF) == (_DWORD)++v57 )
          goto LABEL_91;
      }
    }
    else
    {
LABEL_91:
      v58 = -1;
    }
    sub_B48BF0(v55 - 24, v58, 1);
  }
  sub_2CC0800(*(_QWORD *)(a1 + 176), *(_QWORD *)(a1 + 160), v71, v77, v69, v70, v67, v68, v83);
  nullsub_61();
  v104 = &unk_49DA100;
  nullsub_63();
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  if ( v86 )
    sub_B91220((__int64)&v86, v86);
  return v7;
}
