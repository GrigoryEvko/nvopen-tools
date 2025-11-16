// Function: sub_B01B40
// Address: 0xb01b40
//
_QWORD *__fastcall sub_B01B40(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // r13d
  __int64 v8; // r12
  unsigned __int8 v9; // al
  unsigned __int8 **v10; // rdx
  unsigned __int8 *v11; // rdx
  unsigned __int8 v12; // al
  unsigned __int8 **v13; // rdi
  __int64 v14; // rsi
  _QWORD *v15; // r8
  int v16; // ecx
  __int64 *v17; // r10
  int v18; // r11d
  unsigned int i; // eax
  __int64 *v20; // r9
  __int64 v21; // r15
  unsigned int v22; // ecx
  __int64 v23; // rcx
  unsigned __int8 v24; // al
  __int64 v25; // r13
  __int64 v26; // r12
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  bool v30; // bl
  int v31; // eax
  _BYTE *v32; // rax
  __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rbx
  unsigned __int8 v38; // al
  __int64 v39; // r14
  unsigned __int8 **v40; // rdx
  unsigned __int8 *v41; // r13
  unsigned __int8 v42; // al
  unsigned __int8 **v43; // rdx
  unsigned __int8 **v44; // rdx
  unsigned __int8 v45; // al
  unsigned __int8 *v46; // r13
  unsigned __int8 v47; // al
  unsigned __int8 **v48; // r14
  unsigned __int8 *v49; // r14
  __int64 v50; // rcx
  unsigned __int8 **v51; // rax
  __int64 v52; // rax
  unsigned __int8 **v53; // rax
  unsigned __int8 **v54; // rdx
  unsigned __int16 v55; // ax
  unsigned int v56; // edx
  _QWORD *v57; // rax
  __int64 *v58; // rax
  __int64 v60; // r12
  unsigned __int32 v61; // eax
  unsigned __int32 v62; // edi
  unsigned int v63; // r8d
  __int64 *v64; // rax
  unsigned __int8 **v65; // rdx
  unsigned __int8 **v66; // rax
  int v67; // eax
  __int64 v68; // r9
  char v69; // cl
  __int64 v70; // r8
  _QWORD *v71; // rdi
  int v72; // r8d
  int v73; // r10d
  unsigned int j; // eax
  _QWORD *v75; // rdx
  unsigned int v76; // eax
  __int64 v77; // rax
  __int64 *v78; // r14
  unsigned int v79; // esi
  __int64 v80; // rax
  unsigned int v81; // eax
  __int64 v82; // rdx
  unsigned __int8 *v83; // [rsp-10h] [rbp-1F0h]
  __int64 *v85; // [rsp+28h] [rbp-1B8h]
  __int64 v86; // [rsp+30h] [rbp-1B0h]
  _BYTE *v87; // [rsp+38h] [rbp-1A8h]
  _QWORD *v88; // [rsp+40h] [rbp-1A0h]
  _BYTE *v89; // [rsp+48h] [rbp-198h]
  __int64 *v90; // [rsp+58h] [rbp-188h] BYREF
  _BYTE *v91; // [rsp+60h] [rbp-180h] BYREF
  __int64 v92; // [rsp+68h] [rbp-178h]
  _BYTE v93[48]; // [rsp+70h] [rbp-170h] BYREF
  _BYTE *v94; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v95; // [rsp+A8h] [rbp-138h]
  _BYTE v96[48]; // [rsp+B0h] [rbp-130h] BYREF
  unsigned __int8 *v97; // [rsp+E0h] [rbp-100h] BYREF
  unsigned __int8 **v98; // [rsp+E8h] [rbp-F8h]
  __int64 v99; // [rsp+F0h] [rbp-F0h]
  int v100; // [rsp+F8h] [rbp-E8h]
  unsigned __int8 v101; // [rsp+FCh] [rbp-E4h]
  char v102; // [rsp+100h] [rbp-E0h] BYREF
  __m128i v103; // [rsp+140h] [rbp-A0h] BYREF
  _QWORD *v104; // [rsp+150h] [rbp-90h] BYREF
  unsigned int v105; // [rsp+158h] [rbp-88h]
  char v106; // [rsp+1B0h] [rbp-30h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v85 = (__int64 *)(v3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v3 & 4) != 0 )
    v85 = *(__int64 **)(v3 & 0xFFFFFFFFFFFFFFF8LL);
  v103.m128i_i64[0] = 0;
  v91 = v93;
  v92 = 0x600000000LL;
  v95 = 0x600000000LL;
  v4 = (char *)&v104;
  v94 = v96;
  v103.m128i_i64[1] = 1;
  do
  {
    *(_QWORD *)v4 = -4096;
    v4 += 24;
    *((_QWORD *)v4 - 2) = -4096;
  }
  while ( v4 != &v106 );
  v5 = 0;
  v6 = a1;
  v7 = 0;
  while ( 1 )
  {
    v8 = v6 - 16;
    *(_QWORD *)&v91[8 * v5] = v6;
    LODWORD(v92) = v92 + 1;
    v9 = *(_BYTE *)(v6 - 16);
    v10 = (v9 & 2) != 0 ? *(unsigned __int8 ***)(v6 - 32) : (unsigned __int8 **)(v8 - 8LL * ((v9 >> 2) & 0xF));
    v11 = sub_AF34D0(*v10);
    v12 = *(_BYTE *)(v6 - 16);
    if ( (v12 & 2) != 0 )
    {
      v13 = 0;
      if ( *(_DWORD *)(v6 - 24) != 2 )
        goto LABEL_10;
      v23 = *(_QWORD *)(v6 - 32);
    }
    else
    {
      v13 = 0;
      if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) != 2 )
        goto LABEL_10;
      v23 = v8 - 8LL * ((v12 >> 2) & 0xF);
    }
    v13 = *(unsigned __int8 ***)(v23 + 8);
LABEL_10:
    v97 = v11;
    v98 = v13;
    v14 = v103.m128i_i8[8] & 1;
    if ( (v103.m128i_i8[8] & 1) != 0 )
    {
      v15 = &v104;
      v16 = 3;
      goto LABEL_12;
    }
    v22 = v105;
    v15 = v104;
    if ( !v105 )
    {
      v61 = v103.m128i_u32[2];
      ++v103.m128i_i64[0];
      v90 = 0;
      v62 = ((unsigned __int32)v103.m128i_i32[2] >> 1) + 1;
LABEL_92:
      v63 = 3 * v22;
LABEL_93:
      if ( v63 <= 4 * v62 )
      {
        v79 = 2 * v22;
      }
      else
      {
        v14 = v22 - v103.m128i_i32[3] - v62;
        if ( (unsigned int)v14 > v22 >> 3 )
          goto LABEL_95;
        v79 = v22;
      }
      sub_AFFC00(&v103, v79);
      v14 = (__int64)&v97;
      sub_AF6BB0((__int64)&v103, (__int64 *)&v97, &v90);
      v11 = v97;
      v61 = v103.m128i_u32[2];
LABEL_95:
      v103.m128i_i32[2] = (2 * (v61 >> 1) + 2) | v61 & 1;
      v64 = v90;
      if ( *v90 != -4096 || v90[1] != -4096 )
        --v103.m128i_i32[3];
      *v90 = (__int64)v11;
      v65 = v98;
      *((_DWORD *)v64 + 4) = v7;
      v64[1] = (__int64)v65;
      goto LABEL_28;
    }
    v16 = v105 - 1;
LABEL_12:
    v17 = 0;
    v18 = 1;
    for ( i = v16
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; i = v16 & v81 )
    {
      v20 = &v15[3 * i];
      v21 = *v20;
      if ( v11 == (unsigned __int8 *)*v20 && v13 == (unsigned __int8 **)v20[1] )
        break;
      if ( v21 == -4096 )
      {
        if ( v20[1] == -4096 )
        {
          v61 = v103.m128i_u32[2];
          if ( v17 )
            v20 = v17;
          ++v103.m128i_i64[0];
          v90 = v20;
          v62 = ((unsigned __int32)v103.m128i_i32[2] >> 1) + 1;
          if ( (_BYTE)v14 )
          {
            v63 = 12;
            v22 = 4;
            goto LABEL_93;
          }
          v22 = v105;
          goto LABEL_92;
        }
      }
      else if ( v21 == -8192 && v20[1] == -8192 && !v17 )
      {
        v17 = &v15[3 * i];
      }
      v81 = v18 + i;
      ++v18;
    }
LABEL_28:
    v24 = *(_BYTE *)(v6 - 16);
    if ( (v24 & 2) != 0 )
    {
      if ( *(_DWORD *)(v6 - 24) != 2 )
        break;
      v60 = *(_QWORD *)(v6 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) != 2 )
        break;
      v60 = v8 - 8LL * ((v24 >> 2) & 0xF);
    }
    v6 = *(_QWORD *)(v60 + 8);
    ++v7;
    if ( !v6 )
      break;
    v5 = (unsigned int)v92;
    if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
    {
      sub_C8D5F0(&v91, v93, (unsigned int)v92 + 1LL, 8);
      v5 = (unsigned int)v92;
    }
  }
  v25 = (__int64)v91;
  v26 = (__int64)v94;
  v27 = 0;
  if ( !a2 )
    goto LABEL_77;
  while ( 1 )
  {
    v28 = (unsigned int)v95;
    v29 = (unsigned int)v95 + 1LL;
    if ( v29 > HIDWORD(v95) )
    {
      v14 = (__int64)v96;
      sub_C8D5F0(&v94, v96, v29, 8);
      v28 = (unsigned int)v95;
    }
    *(_QWORD *)&v94[8 * v28] = a2;
    LODWORD(v95) = v95 + 1;
    if ( (_BYTE *)v25 == v91 )
      break;
    v30 = (*(_BYTE *)(a2 - 16) & 2) != 0;
LABEL_35:
    if ( v30 )
      v31 = *(_DWORD *)(a2 - 24);
    else
      v31 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    if ( v31 == 2 )
    {
      v27 = (unsigned int)(v27 + 1);
      a2 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1);
      if ( a2 )
        continue;
    }
    v32 = v91;
    goto LABEL_40;
  }
  v66 = (unsigned __int8 **)sub_A17150((_BYTE *)(a2 - 16));
  v14 = (__int64)sub_AF34D0(*v66);
  v30 = (*(_BYTE *)(a2 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v67 = *(_DWORD *)(a2 - 24);
  else
    v67 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  v68 = 0;
  if ( v67 == 2 )
    v68 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1);
  v69 = v103.m128i_i8[8] & 1;
  if ( (v103.m128i_i8[8] & 1) != 0 )
  {
    v71 = &v104;
    v72 = 3;
    goto LABEL_105;
  }
  v70 = v105;
  v71 = v104;
  if ( v105 )
  {
    v72 = v105 - 1;
LABEL_105:
    v73 = 1;
    for ( j = v72
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4)
                | ((unsigned __int64)(((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4)))); ; j = v72 & v76 )
    {
      v75 = &v71[3 * j];
      if ( v14 == *v75 && v68 == v75[1] )
        break;
      if ( *v75 == -4096 && v75[1] == -4096 )
      {
        if ( !v69 )
        {
          v70 = v105;
          goto LABEL_148;
        }
        v82 = 12;
        goto LABEL_149;
      }
      v76 = v73 + j;
      ++v73;
    }
  }
  else
  {
LABEL_148:
    v82 = 3 * v70;
LABEL_149:
    v75 = &v71[v82];
  }
  v80 = 12;
  if ( !v69 )
    v80 = 3LL * v105;
  if ( v75 == &v71[v80] )
    goto LABEL_35;
  v32 = v91;
  v25 = (__int64)&v91[8 * *((unsigned int *)v75 + 4) + 8];
  v26 = (__int64)&v94[8 * v27 + 8];
LABEL_40:
  if ( (_BYTE *)v25 == v32 )
    goto LABEL_77;
  v33 = *(_QWORD *)(v25 - 8);
  if ( (*(_BYTE *)(v33 - 16) & 2) != 0 )
    v34 = *(_DWORD *)(v33 - 24);
  else
    v34 = (*(_WORD *)(v33 - 16) >> 6) & 0xF;
  v88 = 0;
  if ( v34 == 2 )
    v88 = (_QWORD *)*((_QWORD *)sub_A17150((_BYTE *)(v33 - 16)) + 1);
  v87 = (_BYTE *)v26;
  v35 = v26;
  v89 = (_BYTE *)v25;
  while ( v94 != v87 )
  {
    v36 = *(_QWORD *)(v35 - 8);
    v37 = *((_QWORD *)v89 - 1);
    v38 = *(_BYTE *)(v37 - 16);
    v39 = v37 - 16;
    if ( v36 == v37 )
    {
      if ( (*(_BYTE *)(v37 - 16) & 2) != 0 )
        v78 = *(__int64 **)(v37 - 32);
      else
        v78 = (__int64 *)(v39 - 8LL * ((v38 >> 2) & 0xF));
      v57 = sub_B01860(v85, *(_DWORD *)(v37 + 4), *(unsigned __int16 *)(v37 + 2), *v78, (__int64)v88, 0, 0, 1);
      v14 = (__int64)v83;
      goto LABEL_73;
    }
    if ( (*(_BYTE *)(v37 - 16) & 2) != 0 )
      v40 = *(unsigned __int8 ***)(v37 - 32);
    else
      v40 = (unsigned __int8 **)(v39 - 8LL * ((v38 >> 2) & 0xF));
    v41 = sub_AF34D0(*v40);
    v86 = v36 - 16;
    v42 = *(_BYTE *)(v36 - 16);
    if ( (v42 & 2) != 0 )
      v43 = *(unsigned __int8 ***)(v36 - 32);
    else
      v43 = (unsigned __int8 **)(v86 - 8LL * ((v42 >> 2) & 0xF));
    if ( v41 != sub_AF34D0(*v43) )
      break;
    v45 = *(_BYTE *)(v36 - 16);
    if ( (v45 & 2) != 0 )
    {
      v46 = **(unsigned __int8 ***)(v36 - 32);
      v47 = *(_BYTE *)(v37 - 16);
      if ( (v47 & 2) == 0 )
        goto LABEL_128;
    }
    else
    {
      v44 = (unsigned __int8 **)(8LL * ((v45 >> 2) & 0xF));
      v46 = *(unsigned __int8 **)(v86 - (_QWORD)v44);
      v47 = *(_BYTE *)(v37 - 16);
      if ( (v47 & 2) == 0 )
      {
LABEL_128:
        v48 = (unsigned __int8 **)(v39 - 8LL * ((v47 >> 2) & 0xF));
        goto LABEL_55;
      }
    }
    v48 = *(unsigned __int8 ***)(v37 - 32);
LABEL_55:
    v49 = *v48;
    v97 = 0;
    v99 = 8;
    v98 = (unsigned __int8 **)&v102;
    v100 = 0;
    v101 = 1;
    if ( v49 )
    {
      v50 = 1;
      while ( 1 )
      {
        if ( !(_BYTE)v50 )
          goto LABEL_119;
        v51 = v98;
        v14 = HIDWORD(v99);
        v44 = &v98[HIDWORD(v99)];
        if ( v98 != v44 )
        {
          while ( v49 != *v51 )
          {
            if ( v44 == ++v51 )
              goto LABEL_120;
          }
          goto LABEL_62;
        }
LABEL_120:
        if ( HIDWORD(v99) < (unsigned int)v99 )
        {
          v14 = (unsigned int)++HIDWORD(v99);
          *v44 = v49;
          v50 = v101;
          ++v97;
        }
        else
        {
LABEL_119:
          v14 = (__int64)v49;
          sub_C8CC70(&v97, v49);
          v50 = v101;
        }
LABEL_62:
        if ( *v49 != 18 )
        {
          v52 = sub_AF2660(v49);
          v50 = v101;
          v49 = (unsigned __int8 *)v52;
          if ( v52 )
            continue;
        }
        if ( v46 )
          goto LABEL_65;
        goto LABEL_117;
      }
    }
    if ( !v46 )
      goto LABEL_70;
    v50 = 1;
    while ( 1 )
    {
LABEL_65:
      if ( !(_BYTE)v50 )
      {
        v14 = (__int64)v46;
        if ( sub_C8CA60(&v97, v46, v44, v50) )
        {
          LOBYTE(v50) = v101;
          goto LABEL_117;
        }
        goto LABEL_115;
      }
      v53 = v98;
      v54 = &v98[HIDWORD(v99)];
      if ( v98 != v54 )
        break;
LABEL_115:
      if ( *v46 == 18 )
      {
        LOBYTE(v50) = v101;
        v46 = 0;
        goto LABEL_117;
      }
      v77 = sub_AF2660(v46);
      v50 = v101;
      v46 = (unsigned __int8 *)v77;
      if ( !v77 )
      {
LABEL_117:
        if ( !(_BYTE)v50 )
          _libc_free(v98, v14);
        goto LABEL_70;
      }
    }
    while ( *v53 != v46 )
    {
      if ( v54 == ++v53 )
        goto LABEL_115;
    }
LABEL_70:
    v14 = *(unsigned int *)(v37 + 4);
    v55 = *(_WORD *)(v36 + 2);
    if ( (_DWORD)v14 == *(_DWORD *)(v36 + 4) )
    {
      v56 = v55;
      if ( *(_WORD *)(v37 + 2) != v55 )
        v56 = 0;
    }
    else
    {
      v14 = 0;
      v56 = 0;
    }
    v57 = sub_B01860(v85, v14, v56, (__int64)v46, (__int64)v88, 0, 0, 1);
LABEL_73:
    if ( !v57 )
      break;
    v89 -= 8;
    v87 -= 8;
    v88 = v57;
    if ( v91 == v89 )
      goto LABEL_78;
    v35 = (__int64)v87;
  }
  if ( !v88 )
  {
LABEL_77:
    v58 = (__int64 *)sub_A17150((_BYTE *)(a1 - 16));
    v14 = 0;
    v88 = sub_B01860(v85, 0, 0, *v58, 0, 0, 0, 1);
  }
LABEL_78:
  if ( (v103.m128i_i8[8] & 1) == 0 )
  {
    v14 = 24LL * v105;
    sub_C7D6A0(v104, v14, 8);
  }
  if ( v94 != v96 )
    _libc_free(v94, v14);
  if ( v91 != v93 )
    _libc_free(v91, v14);
  return v88;
}
