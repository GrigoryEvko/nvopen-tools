// Function: sub_1B57F80
// Address: 0x1b57f80
//
__int64 __fastcall sub_1B57F80(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, _BYTE *a6, __int64 a7)
{
  __m128i *v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 **v10; // r13
  __int64 *v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rdx
  int v17; // eax
  __int64 **v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rdx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rax
  __int64 *v27; // rax
  unsigned int v28; // eax
  char v29; // al
  __int64 v31; // r14
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r14
  __int64 v35; // r15
  _QWORD *v36; // r12
  unsigned __int8 v37; // al
  unsigned __int64 v38; // rax
  _QWORD *v39; // rcx
  __int64 *v40; // rax
  __int64 v41; // rdi
  _QWORD *v42; // rsi
  int v43; // ecx
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // r8
  __int64 *v47; // r10
  __int64 v48; // r15
  __int64 v49; // rbx
  __int64 **v50; // r8
  __int64 v51; // rax
  _QWORD *v52; // rsi
  int v53; // ecx
  unsigned int v54; // edx
  __int64 *v55; // rax
  __int64 *v56; // rsi
  __int64 v57; // rax
  _QWORD *v58; // rsi
  int v59; // ecx
  unsigned int v60; // edx
  __int64 v61; // rdi
  int j; // eax
  int v63; // r8d
  int v64; // eax
  int v65; // edi
  _QWORD *v66; // rdi
  int v67; // esi
  unsigned int v68; // ecx
  __int64 *v69; // rdx
  __int64 v70; // r11
  int i; // eax
  int v72; // r8d
  int v73; // edx
  int v74; // r8d
  int k; // eax
  int v76; // edi
  __int64 **v77; // [rsp+0h] [rbp-1F0h]
  __int64 *v78; // [rsp+8h] [rbp-1E8h]
  __int64 *v79; // [rsp+8h] [rbp-1E8h]
  __int64 v83; // [rsp+48h] [rbp-1A8h]
  __int64 v85; // [rsp+68h] [rbp-188h]
  __int64 v86; // [rsp+70h] [rbp-180h] BYREF
  __int64 v87; // [rsp+78h] [rbp-178h] BYREF
  __m128i v88; // [rsp+80h] [rbp-170h]
  _BYTE v89[16]; // [rsp+90h] [rbp-160h] BYREF
  void (__fastcall *v90)(__int64 **, __int64 **, __int64); // [rsp+A0h] [rbp-150h]
  unsigned __int8 (__fastcall *v91)(__int64 **, __int64 *); // [rsp+A8h] [rbp-148h]
  __int64 *v92; // [rsp+B0h] [rbp-140h]
  __int64 v93; // [rsp+B8h] [rbp-138h]
  _BYTE v94[16]; // [rsp+C0h] [rbp-130h] BYREF
  void (__fastcall *v95)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-120h]
  __int64 *v96; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v97; // [rsp+E8h] [rbp-108h] BYREF
  _BYTE v98[32]; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v99; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v100; // [rsp+118h] [rbp-D8h]
  __int64 *v101; // [rsp+120h] [rbp-D0h] BYREF
  int v102; // [rsp+128h] [rbp-C8h]
  __m128i v103; // [rsp+160h] [rbp-90h] BYREF
  _BYTE v104[16]; // [rsp+170h] [rbp-80h] BYREF
  void (__fastcall *v105)(_BYTE *, _BYTE *, __int64); // [rsp+180h] [rbp-70h]
  __int64 *v106; // [rsp+190h] [rbp-60h]
  __int64 v107; // [rsp+198h] [rbp-58h]
  _BYTE v108[16]; // [rsp+1A0h] [rbp-50h] BYREF
  void (__fastcall *v109)(_BYTE *, _BYTE *, __int64); // [rsp+1B0h] [rbp-40h]

  v83 = *(_QWORD *)(a1 + 40);
  v7 = (__m128i *)&v101;
  v99 = 0;
  v100 = 1;
  do
  {
    v7->m128i_i64[0] = -8;
    ++v7;
  }
  while ( v7 != &v103 );
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v8 = *(__int64 **)(a1 - 8);
  else
    v8 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v9 = *v8;
  v97 = a2;
  v10 = (__int64 **)v89;
  v96 = (__int64 *)v9;
  sub_1B57D30((__int64)&v103, (__int64)&v99, (__int64 *)&v96, &v97);
  sub_1580910(&v103);
  v88 = v103;
  sub_1974F30((__int64)v89, (__int64)v104);
  v11 = (__int64 *)v108;
  v92 = v106;
  v93 = v107;
  sub_1974F30((__int64)v94, (__int64)v108);
  v14 = v88.m128i_i64[0];
  if ( v92 == (__int64 *)v88.m128i_i64[0] )
    goto LABEL_21;
  while ( 1 )
  {
    if ( !v14 )
      BUG();
    v15 = v14 - 24;
    v16 = *(unsigned __int8 *)(v14 - 8);
    if ( (unsigned int)(v16 - 25) <= 9 )
    {
      if ( (unsigned int)sub_15F4D60(v14 - 24) != 1 )
        goto LABEL_37;
      v17 = *(unsigned __int8 *)(v14 - 8);
      if ( (unsigned int)(v17 - 24) > 6 )
      {
        if ( (unsigned int)(v17 - 32) <= 2 )
          goto LABEL_37;
      }
      else if ( (unsigned int)(v17 - 24) > 4 )
      {
        goto LABEL_37;
      }
      v11 = 0;
      v18 = (__int64 **)(v14 - 24);
      v19 = sub_15F4DF0(v14 - 24, 0);
      v12 = a3;
      a3 = v19;
      v83 = v12;
      goto LABEL_12;
    }
    if ( (_BYTE)v16 != 79 )
      break;
    v31 = *(_QWORD *)(v14 - 96);
    if ( *(_BYTE *)(v31 + 16) > 0x10u )
    {
      if ( (v100 & 1) != 0 )
      {
        v11 = (__int64 *)&v101;
        v12 = 3;
      }
      else
      {
        v11 = v101;
        if ( !v102 )
          goto LABEL_21;
        v12 = (unsigned int)(v102 - 1);
      }
      v16 = (unsigned int)v12 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v40 = &v11[2 * v16];
      v41 = *v40;
      if ( v31 != *v40 )
      {
        for ( i = 1; ; i = v72 )
        {
          if ( v41 == -8 )
            goto LABEL_21;
          v72 = i + 1;
          v16 = (unsigned int)v12 & (i + (_DWORD)v16);
          v40 = &v11[2 * (unsigned int)v16];
          v41 = *v40;
          if ( v31 == *v40 )
            break;
        }
      }
      v31 = v40[1];
      if ( !v31 )
        goto LABEL_21;
    }
    if ( sub_1596070(v31, (__int64)v11, v16, v12) )
    {
      v34 = *(_QWORD *)(v14 - 72);
      if ( *(_BYTE *)(v34 + 16) > 0x10u )
      {
        if ( (v100 & 1) != 0 )
        {
          v58 = &v101;
          v59 = 3;
        }
        else
        {
          v58 = v101;
          if ( !v102 )
            goto LABEL_21;
          v59 = v102 - 1;
        }
        v60 = v59 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v45 = &v58[2 * v60];
        v61 = *v45;
        if ( v34 != *v45 )
        {
          for ( j = 1; ; j = v63 )
          {
            if ( v61 == -8 )
              goto LABEL_21;
            v63 = j + 1;
            v60 = v59 & (j + v60);
            v45 = &v58[2 * v60];
            v61 = *v45;
            if ( v34 == *v45 )
              break;
          }
        }
LABEL_76:
        v34 = v45[1];
        goto LABEL_95;
      }
    }
    else
    {
      if ( !sub_1593BB0(v31, (__int64)v11, v32, v33) )
        goto LABEL_21;
      v34 = *(_QWORD *)(v14 - 48);
      if ( *(_BYTE *)(v34 + 16) > 0x10u )
      {
        if ( (v100 & 1) != 0 )
        {
          v42 = &v101;
          v43 = 3;
        }
        else
        {
          v42 = v101;
          if ( !v102 )
            goto LABEL_21;
          v43 = v102 - 1;
        }
        v44 = v43 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v45 = &v42[2 * v44];
        v46 = *v45;
        if ( v34 != *v45 )
        {
          for ( k = 1; ; k = v76 )
          {
            if ( v46 == -8 )
              goto LABEL_21;
            v76 = k + 1;
            v44 = v43 & (k + v44);
            v45 = &v42[2 * v44];
            v46 = *v45;
            if ( v34 == *v45 )
              break;
          }
        }
        goto LABEL_76;
      }
    }
LABEL_54:
    v35 = *(_QWORD *)(v14 - 16);
    if ( v35 )
    {
      while ( 1 )
      {
        v36 = sub_1648700(v35);
        v37 = *((_BYTE *)v36 + 16);
        if ( v37 <= 0x17u )
          break;
        if ( v36[5] != a3 )
        {
          if ( v37 != 77 )
            break;
          v38 = 0xAAAAAAAAAAAAAAABLL * ((v35 - sub_13CF970((__int64)v36)) >> 3);
          v39 = (*((_BYTE *)v36 + 23) & 0x40) != 0
              ? (_QWORD *)*(v36 - 1)
              : &v36[-3 * (*((_DWORD *)v36 + 5) & 0xFFFFFFF)];
          if ( a3 != v39[3 * *((unsigned int *)v36 + 14) + 1 + (unsigned int)v38] )
            break;
        }
        v35 = *(_QWORD *)(v35 + 8);
        if ( !v35 )
          goto LABEL_62;
      }
LABEL_37:
      if ( v95 )
        v95(v94, v94, 3);
      if ( v90 )
        v90(v10, v10, 3);
      if ( v109 )
        v109(v108, v108, 3);
      if ( v105 )
        v105(v104, v104, 3);
      goto LABEL_45;
    }
LABEL_62:
    v11 = &v99;
    v18 = &v96;
    v86 = v15;
    v87 = v34;
    sub_1B57D30((__int64)&v96, (__int64)&v99, &v86, &v87);
LABEL_12:
    v14 = *(_QWORD *)(v88.m128i_i64[0] + 8);
    v88.m128i_i64[0] = v14;
    if ( v14 == v88.m128i_i64[1] )
    {
LABEL_20:
      if ( v92 == (__int64 *)v14 )
        goto LABEL_21;
    }
    else
    {
      v11 = (__int64 *)v14;
      do
      {
        if ( v11 )
          v11 -= 3;
        if ( !v90 )
          sub_4263D6(v18, v11, v20);
        v18 = v10;
        if ( v91(v10, v11) )
        {
          v14 = v88.m128i_i64[0];
          goto LABEL_20;
        }
        v11 = *(__int64 **)(v88.m128i_i64[0] + 8);
        v88.m128i_i64[0] = (__int64)v11;
      }
      while ( (__int64 *)v88.m128i_i64[1] != v11 );
      v14 = (__int64)v11;
      if ( v92 == v11 )
        goto LABEL_21;
    }
  }
  v47 = (__int64 *)v98;
  v96 = (__int64 *)v98;
  v97 = 0x400000000LL;
  if ( (*(_DWORD *)(v14 - 4) & 0xFFFFFFF) == 0 )
  {
    v56 = (__int64 *)v98;
    goto LABEL_98;
  }
  v48 = 0;
  v49 = 24LL * (*(_DWORD *)(v14 - 4) & 0xFFFFFFF);
  v50 = v10;
  do
  {
    if ( (*(_BYTE *)(v14 - 1) & 0x40) != 0 )
    {
      v34 = *(_QWORD *)(*(_QWORD *)(v14 - 32) + v48);
      if ( *(_BYTE *)(v34 + 16) <= 0x10u )
        goto LABEL_80;
    }
    else
    {
      v34 = *(_QWORD *)(v14 - 24 - 24LL * (*(_DWORD *)(v14 - 4) & 0xFFFFFFF) + v48);
      if ( *(_BYTE *)(v34 + 16) <= 0x10u )
        goto LABEL_80;
    }
    if ( (v100 & 1) != 0 )
    {
      v52 = &v101;
      v53 = 3;
    }
    else
    {
      v52 = v101;
      if ( !v102 )
        goto LABEL_92;
      v53 = v102 - 1;
    }
    v54 = v53 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v55 = &v52[2 * v54];
    v13 = *v55;
    if ( v34 != *v55 )
    {
      v64 = 1;
      while ( v13 != -8 )
      {
        v65 = v64 + 1;
        v54 = v53 & (v64 + v54);
        v55 = &v52[2 * v54];
        v13 = *v55;
        if ( v34 == *v55 )
          goto LABEL_88;
        v64 = v65;
      }
LABEL_92:
      v15 = v14 - 24;
      v34 = 0;
      v10 = v50;
      goto LABEL_93;
    }
LABEL_88:
    v34 = v55[1];
    if ( !v34 )
    {
      v15 = v14 - 24;
      v10 = v50;
      goto LABEL_93;
    }
LABEL_80:
    v51 = (unsigned int)v97;
    if ( (unsigned int)v97 >= HIDWORD(v97) )
    {
      v77 = v50;
      v78 = v47;
      sub_16CD150((__int64)&v96, v47, 0, 8, (int)v50, v13);
      v51 = (unsigned int)v97;
      v50 = v77;
      v47 = v78;
    }
    v48 += 24;
    v96[v51] = v34;
    LODWORD(v97) = v97 + 1;
  }
  while ( v49 != v48 );
  v56 = v96;
  v15 = v14 - 24;
  v10 = v50;
LABEL_98:
  v79 = v47;
  if ( (unsigned __int8)(*(_BYTE *)(v14 - 8) - 75) > 1u )
    v57 = sub_14DD1F0(v15, v56, (unsigned int)v97, a6, 0);
  else
    v57 = sub_14D7760(*(_WORD *)(v14 - 6) & 0x7FFF, (_QWORD *)*v56, v56[1], (__int64)a6, 0);
  v47 = v79;
  v34 = v57;
LABEL_93:
  if ( v96 != v47 )
    _libc_free((unsigned __int64)v96);
LABEL_95:
  if ( v34 )
    goto LABEL_54;
LABEL_21:
  sub_A17130((__int64)v94);
  sub_A17130((__int64)v10);
  sub_A17130((__int64)v108);
  sub_A17130((__int64)v104);
  if ( *a4 )
  {
    if ( *a4 == a3 )
      goto LABEL_23;
LABEL_45:
    LODWORD(v14) = 0;
    v29 = v100 & 1;
    goto LABEL_46;
  }
  *a4 = a3;
LABEL_23:
  v103.m128i_i64[0] = sub_157F280(a3);
  v22 = v103.m128i_i64[0];
  if ( v103.m128i_i64[0] == v21 )
  {
LABEL_133:
    LOBYTE(v14) = *(_DWORD *)(a5 + 8) != 0;
    v29 = v100 & 1;
    goto LABEL_46;
  }
  v85 = v21;
  while ( 2 )
  {
    v28 = sub_1B46990(v22, v83);
    if ( v28 == -1 )
      goto LABEL_32;
    if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
      v23 = *(_QWORD *)(v22 - 8);
    else
      v23 = v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
    v14 = *(_QWORD *)(v23 + 24LL * v28);
    if ( !v14 )
      BUG();
    if ( *(_BYTE *)(v14 + 16) <= 0x10u )
      goto LABEL_28;
    v29 = v100 & 1;
    if ( (v100 & 1) != 0 )
    {
      v66 = &v101;
      v67 = 3;
    }
    else
    {
      v66 = v101;
      if ( !v102 )
        goto LABEL_120;
      v67 = v102 - 1;
    }
    v68 = v67 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v69 = &v66[2 * v68];
    v70 = *v69;
    if ( v14 == *v69 )
    {
LABEL_119:
      v14 = v69[1];
      if ( !v14 )
        goto LABEL_120;
LABEL_28:
      if ( !(unsigned __int8)sub_1B42DA0(v14, a7) )
      {
        v29 = v100 & 1;
        goto LABEL_120;
      }
      v26 = *(unsigned int *)(a5 + 8);
      if ( (unsigned int)v26 >= *(_DWORD *)(a5 + 12) )
      {
        sub_16CD150(a5, (const void *)(a5 + 16), 0, 16, v24, v25);
        v26 = *(unsigned int *)(a5 + 8);
      }
      v27 = (__int64 *)(*(_QWORD *)a5 + 16 * v26);
      *v27 = v22;
      v27[1] = v14;
      ++*(_DWORD *)(a5 + 8);
LABEL_32:
      sub_1B42F80((__int64)&v103);
      v22 = v103.m128i_i64[0];
      if ( v85 == v103.m128i_i64[0] )
        goto LABEL_133;
      continue;
    }
    break;
  }
  v73 = 1;
  while ( v70 != -8 )
  {
    v74 = v73 + 1;
    v68 = v67 & (v68 + v73);
    v69 = &v66[2 * v68];
    v70 = *v69;
    if ( v14 == *v69 )
      goto LABEL_119;
    v73 = v74;
  }
LABEL_120:
  LODWORD(v14) = 0;
LABEL_46:
  if ( !v29 )
    j___libc_free_0(v101);
  return (unsigned int)v14;
}
