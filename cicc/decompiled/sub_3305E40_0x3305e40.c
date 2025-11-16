// Function: sub_3305E40
// Address: 0x3305e40
//
__int64 __fastcall sub_3305E40(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdi
  unsigned int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int16 *v8; // rdx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // r15
  int v17; // r13d
  __int128 v18; // kr00_16
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rbx
  char *v22; // rax
  char *v23; // rbx
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // r15
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // r11
  __int64 v34; // rdx
  __int64 v35; // r8
  int v36; // r14d
  char *v38; // r12
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // r15
  __int64 v42; // r14
  __int64 v43; // r9
  int v44; // eax
  int v45; // eax
  int v46; // eax
  char *v47; // rax
  int v48; // eax
  unsigned int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // r11
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 v56; // rax
  unsigned int v57; // edx
  __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // r11
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // rax
  unsigned int v64; // edx
  int v65; // [rsp+0h] [rbp-190h]
  int v66; // [rsp+0h] [rbp-190h]
  int v67; // [rsp+8h] [rbp-188h]
  __int64 v68; // [rsp+8h] [rbp-188h]
  __int64 v69; // [rsp+8h] [rbp-188h]
  __int64 v70; // [rsp+10h] [rbp-180h]
  __int64 v71; // [rsp+10h] [rbp-180h]
  __int64 v72; // [rsp+20h] [rbp-170h]
  __int64 v73; // [rsp+28h] [rbp-168h]
  __int128 v74; // [rsp+30h] [rbp-160h]
  __int64 v75; // [rsp+40h] [rbp-150h]
  __int64 v77; // [rsp+50h] [rbp-140h]
  unsigned int v78; // [rsp+50h] [rbp-140h]
  _BYTE *v79; // [rsp+60h] [rbp-130h]
  __int128 v80; // [rsp+60h] [rbp-130h]
  unsigned __int64 v81; // [rsp+70h] [rbp-120h]
  int v82; // [rsp+70h] [rbp-120h]
  __int64 v83; // [rsp+70h] [rbp-120h]
  __int128 v84; // [rsp+70h] [rbp-120h]
  unsigned __int8 v86; // [rsp+8Fh] [rbp-101h]
  unsigned __int64 v87; // [rsp+98h] [rbp-F8h] BYREF
  unsigned __int16 v88; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v89; // [rsp+A8h] [rbp-E8h]
  __int64 v90; // [rsp+B0h] [rbp-E0h]
  __int64 v91; // [rsp+B8h] [rbp-D8h]
  __int64 v92; // [rsp+C0h] [rbp-D0h] BYREF
  int v93; // [rsp+C8h] [rbp-C8h]
  __int64 v94; // [rsp+D0h] [rbp-C0h]
  int v95; // [rsp+D8h] [rbp-B8h]
  __int64 v96; // [rsp+E0h] [rbp-B0h] BYREF
  char *v97; // [rsp+E8h] [rbp-A8h]
  __int64 v98; // [rsp+F0h] [rbp-A0h]
  int v99; // [rsp+F8h] [rbp-98h]
  char v100; // [rsp+FCh] [rbp-94h]
  char v101; // [rsp+100h] [rbp-90h] BYREF
  _BYTE *v102; // [rsp+110h] [rbp-80h] BYREF
  __int64 v103; // [rsp+118h] [rbp-78h]
  _BYTE v104[112]; // [rsp+120h] [rbp-70h] BYREF

  v2 = *(_QWORD **)(a2 + 40);
  v3 = v2[5];
  v86 = *(_DWORD *)(v3 + 24) == 11 || *(_DWORD *)(v3 + 24) == 35;
  if ( !v86 )
    return v86;
  v4 = *(_QWORD *)(v3 + 96);
  v5 = *(_DWORD *)(v4 + 32);
  if ( v5 > 0x40 )
  {
    v36 = sub_C445E0(v4 + 24);
    if ( !v36 )
      return 0;
    v86 = 0;
    if ( v5 != (unsigned int)sub_C444A0(v4 + 24) + v36 )
      return v86;
  }
  else
  {
    v6 = *(_QWORD *)(v4 + 24);
    if ( !v6 || (v6 & (v6 + 1)) != 0 )
      return 0;
  }
  if ( *(_DWORD *)(*v2 + 24LL) == 298 )
    return 0;
  v96 = 0;
  v102 = v104;
  v103 = 0x800000000LL;
  v97 = &v101;
  v98 = 2;
  v7 = *a1;
  v99 = 0;
  v100 = 1;
  v87 = 0;
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v7 + 544LL) - 42) <= 1 )
  {
    v8 = *(unsigned __int16 **)(a2 + 48);
    v9 = *v8;
    v10 = *((_QWORD *)v8 + 1);
    v88 = v9;
    v89 = v10;
    if ( v9 )
    {
      if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
        BUG();
      v12 = 16LL * (v9 - 1);
      v11 = *(_QWORD *)&byte_444C4A0[v12];
      LOBYTE(v12) = byte_444C4A0[v12 + 8];
    }
    else
    {
      v11 = sub_3007260((__int64)&v88);
      v90 = v11;
      v91 = v12;
    }
    v92 = v11;
    LOBYTE(v93) = v12;
    if ( (unsigned __int64)sub_CA1930(&v92) <= 0x20 )
      goto LABEL_38;
  }
  v86 = sub_32820D0((__int64)a1, a2, (__int64)&v102, (__int64)&v96, v3, (unsigned __int64)&v87);
  if ( v86 && (_DWORD)v103 )
  {
    v13 = v87;
    v74 = (__int128)_mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
    if ( v87 )
    {
      v14 = *(_QWORD *)(v87 + 80);
      v15 = *a1;
      v16 = *(_QWORD *)(*(_QWORD *)(v87 + 48) + 8LL);
      v17 = **(unsigned __int16 **)(v87 + 48);
      v92 = v14;
      v18 = v87;
      if ( v14 )
      {
        v81 = v87;
        sub_B96E90((__int64)&v92, v14, 1);
        v18 = v81;
      }
      v93 = *(_DWORD *)(v13 + 72);
      v19 = sub_3406EB0(v15, 186, (unsigned int)&v92, v17, v16, DWORD2(v18), v18, v74);
      v21 = v20;
      if ( v92 )
        sub_B91220((__int64)&v92, v92);
      sub_34161C0(*a1, v87, 0, v19, v21);
      if ( *(_DWORD *)(v19 + 24) == 186 )
        sub_33EC010(*a1, v19, v87, 0, v74, *((_QWORD *)&v74 + 1));
    }
    v22 = v97;
    if ( v100 )
      v23 = &v97[8 * HIDWORD(v98)];
    else
      v23 = &v97[8 * (unsigned int)v98];
    if ( v97 != v23 )
    {
      while ( *(_QWORD *)v22 >= 0xFFFFFFFFFFFFFFFELL )
      {
        v22 += 8;
        if ( v23 == v22 )
          goto LABEL_24;
      }
      if ( v22 != v23 )
      {
        v38 = v22;
        v39 = *(_QWORD *)v22;
        do
        {
          v40 = *(_QWORD *)(v39 + 40);
          v41 = *(_QWORD *)v40;
          v42 = *(unsigned int *)(v40 + 8);
          v43 = *(_QWORD *)(v40 + 40);
          v84 = (__int128)_mm_loadu_si128((const __m128i *)v40);
          v80 = (__int128)_mm_loadu_si128((const __m128i *)(v40 + 40));
          v78 = *(_DWORD *)(v40 + 48);
          v44 = *(_DWORD *)(*(_QWORD *)v40 + 24LL);
          if ( v44 == 35 || v44 == 11 )
          {
            v58 = *(_QWORD *)(v41 + 80);
            v59 = *(_QWORD *)(v41 + 48) + 16LL * (unsigned int)v42;
            v60 = *a1;
            v61 = v72;
            LOWORD(v61) = *(_WORD *)v59;
            v62 = *(_QWORD *)(v59 + 8);
            v92 = v58;
            v72 = v61;
            if ( v58 )
            {
              v66 = v60;
              v69 = v43;
              sub_B96E90((__int64)&v92, v58, 1);
              LODWORD(v60) = v66;
              v43 = v69;
            }
            v71 = v43;
            v93 = *(_DWORD *)(v41 + 72);
            v63 = sub_3406EB0(v60, 186, (unsigned int)&v92, v72, v62, v43, v84, v74);
            v43 = v71;
            v41 = v63;
            v42 = v64;
            if ( v92 )
            {
              sub_B91220((__int64)&v92, v92);
              v43 = v71;
            }
          }
          v45 = *(_DWORD *)(v43 + 24);
          if ( v45 == 35 || v45 == 11 )
          {
            v51 = *(_QWORD *)(v43 + 80);
            v52 = *(_QWORD *)(v43 + 48) + 16LL * v78;
            v53 = *a1;
            v54 = v73;
            LOWORD(v54) = *(_WORD *)v52;
            v55 = *(_QWORD *)(v52 + 8);
            v92 = v51;
            v73 = v54;
            if ( v51 )
            {
              v65 = v53;
              v67 = v55;
              v70 = v43;
              sub_B96E90((__int64)&v92, v51, 1);
              LODWORD(v53) = v65;
              LODWORD(v55) = v67;
              v43 = v70;
            }
            v93 = *(_DWORD *)(v43 + 72);
            v56 = sub_3406EB0(v53, 186, (unsigned int)&v92, v73, v55, v43, v80, v74);
            v78 = v57;
            v43 = v56;
            if ( v92 )
            {
              v68 = v56;
              sub_B91220((__int64)&v92, v92);
              v43 = v68;
            }
          }
          v46 = *(_DWORD *)(v41 + 24);
          if ( v46 == 35 || v46 == 11 )
          {
            v48 = *(_DWORD *)(v43 + 24);
            if ( v48 != 11 && v48 != 35 )
            {
              v49 = v42;
              v42 = v78;
              v78 = v49;
              v50 = v41;
              v41 = v43;
              v43 = v50;
            }
          }
          sub_33EC010(
            *a1,
            v39,
            v41,
            v42 | *((_QWORD *)&v84 + 1) & 0xFFFFFFFF00000000LL,
            v43,
            v78 | *((_QWORD *)&v80 + 1) & 0xFFFFFFFF00000000LL);
          v47 = v38 + 8;
          if ( v38 + 8 == v23 )
            break;
          v39 = *(_QWORD *)v47;
          for ( v38 += 8; *(_QWORD *)v47 >= 0xFFFFFFFFFFFFFFFELL; v38 = v47 )
          {
            v47 += 8;
            if ( v23 == v47 )
              goto LABEL_24;
            v39 = *(_QWORD *)v47;
          }
        }
        while ( v23 != v38 );
      }
    }
LABEL_24:
    v24 = (unsigned __int64)v102;
    v79 = &v102[8 * (unsigned int)v103];
    if ( v79 != v102 )
    {
      do
      {
        v27 = *(_QWORD *)v24;
        v28 = *a1;
        v29 = *(_QWORD *)(*(_QWORD *)v24 + 48LL);
        v30 = *(_QWORD *)(*(_QWORD *)v24 + 80LL);
        v31 = *(_QWORD *)v24;
        v32 = v75;
        LOWORD(v32) = *(_WORD *)v29;
        v33 = *(_QWORD *)(v29 + 8);
        v92 = v30;
        v75 = v32;
        if ( v30 )
        {
          v82 = v33;
          sub_B96E90((__int64)&v92, v30, 1);
          v31 = v27;
          LODWORD(v33) = v82;
        }
        v93 = *(_DWORD *)(v27 + 72);
        v83 = sub_3406EB0(v28, 186, (unsigned int)&v92, v75, v33, 0, (unsigned __int64)v31, v74);
        v35 = v34;
        if ( v92 )
        {
          v77 = v34;
          sub_B91220((__int64)&v92, v92);
          v35 = v77;
        }
        sub_34161C0(*a1, v27, 0, v83, v35);
        if ( *(_DWORD *)(v83 + 24) == 186 )
          v83 = sub_33EC010(*a1, v83, v27, 0, v74, *((_QWORD *)&v74 + 1));
        v24 += 8LL;
        v25 = sub_32B3F40(a1, v83);
        v93 = v26;
        v92 = v25;
        v94 = v25;
        v95 = 1;
        sub_32EB790((__int64)a1, v27, &v92, 2, 1);
      }
      while ( v79 != (_BYTE *)v24 );
    }
    sub_34158F0(*a1, a2, **(_QWORD **)(a2 + 40));
  }
  else
  {
LABEL_38:
    v86 = 0;
  }
  if ( !v100 )
    _libc_free((unsigned __int64)v97);
  if ( v102 != v104 )
    _libc_free((unsigned __int64)v102);
  return v86;
}
