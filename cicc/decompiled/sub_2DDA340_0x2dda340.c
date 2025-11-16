// Function: sub_2DDA340
// Address: 0x2dda340
//
__int64 __fastcall sub_2DDA340(__int64 a1, __int64 a2)
{
  __m128i *v3; // rsi
  __m128i *v4; // r15
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned __int8 **v12; // rbx
  __int64 v13; // r13
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int8 *v19; // rax
  __int64 v20; // rcx
  unsigned __int8 **v21; // rdx
  unsigned __int8 *v22; // r12
  unsigned __int8 **v23; // rbx
  unsigned __int8 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 i; // r15
  __int64 v30; // rax
  __int64 v31; // r13
  const char *v32; // rsi
  __int64 v33; // rdx
  unsigned __int8 v34; // al
  const char *v35; // rax
  unsigned __int64 v36; // rdx
  const char *v37; // rax
  unsigned __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rcx
  __m128i v43; // rax
  __int64 *v44; // rdi
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r12
  __int64 v48; // rax
  unsigned int *v49; // rbx
  unsigned int *v50; // r13
  unsigned int v51; // r12d
  unsigned __int64 v52; // r13
  _BYTE *v53; // rbx
  unsigned int *v54; // rbx
  unsigned int *v55; // r14
  _BYTE *v56; // rbx
  unsigned __int64 v57; // rdi
  unsigned int *v58; // rbx
  unsigned int *v59; // r13
  unsigned __int64 v60; // rdi
  unsigned int *v61; // rbx
  unsigned int *v62; // r13
  unsigned __int64 v63; // rdi
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 v67; // rdi
  unsigned int v68; // eax
  __int64 v69; // rcx
  __int64 v70; // [rsp+0h] [rbp-160h]
  __int64 v72; // [rsp+20h] [rbp-140h]
  __m128i *v73; // [rsp+28h] [rbp-138h]
  __int64 v74; // [rsp+28h] [rbp-138h]
  char v76; // [rsp+38h] [rbp-128h]
  unsigned __int8 **v77; // [rsp+48h] [rbp-118h]
  __int32 v78; // [rsp+48h] [rbp-118h]
  __m128i *v79; // [rsp+50h] [rbp-110h]
  unsigned __int64 v80; // [rsp+50h] [rbp-110h]
  unsigned __int8 **v81; // [rsp+58h] [rbp-108h]
  __int64 v82; // [rsp+58h] [rbp-108h]
  __int64 v83; // [rsp+60h] [rbp-100h] BYREF
  __int64 v84; // [rsp+68h] [rbp-F8h] BYREF
  unsigned __int64 v85; // [rsp+70h] [rbp-F0h] BYREF
  __int8 v86; // [rsp+78h] [rbp-E8h]
  __m128i v87; // [rsp+80h] [rbp-E0h] BYREF
  unsigned __int64 v88; // [rsp+90h] [rbp-D0h]
  __int64 v89; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+A8h] [rbp-B8h]
  __int64 v91; // [rsp+B0h] [rbp-B0h]
  unsigned int v92; // [rsp+B8h] [rbp-A8h]
  unsigned int *v93; // [rsp+C0h] [rbp-A0h]
  __int64 v94; // [rsp+C8h] [rbp-98h]
  __int64 v95; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v96; // [rsp+D8h] [rbp-88h]
  __int64 v97; // [rsp+E0h] [rbp-80h]
  unsigned int v98; // [rsp+E8h] [rbp-78h]
  unsigned int *v99; // [rsp+F0h] [rbp-70h]
  __int64 v100; // [rsp+F8h] [rbp-68h]
  __int64 v101; // [rsp+100h] [rbp-60h] BYREF
  __int64 v102; // [rsp+108h] [rbp-58h]
  __int64 v103; // [rsp+110h] [rbp-50h]
  unsigned int v104; // [rsp+118h] [rbp-48h]
  _BYTE *v105; // [rsp+120h] [rbp-40h]
  __int64 v106; // [rsp+128h] [rbp-38h]
  _BYTE v107[48]; // [rsp+130h] [rbp-30h] BYREF

  *(_BYTE *)(a1 + 24) = *(_DWORD *)(a2 + 284) == 5;
  v70 = a2 + 312;
  v93 = (unsigned int *)&v95;
  v99 = (unsigned int *)&v101;
  v105 = v107;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v106 = 0;
  sub_2DD9E60(a1, a2, (__int64)"llvm.used", 9u);
  sub_2DD9E60(a1, a2, (__int64)"llvm.compiler.used", 0x12u);
  v3 = *(__m128i **)(a2 + 32);
  v73 = v3;
  v72 = a2 + 24;
  if ( v3 != (__m128i *)(a2 + 24) )
  {
    while ( 1 )
    {
      if ( !v73 )
        BUG();
      v4 = (__m128i *)v73[1].m128i_i64[1];
      v79 = v73 + 1;
      if ( v4 != &v73[1] )
        break;
LABEL_34:
      v73 = (__m128i *)v73->m128i_i64[1];
      if ( (__m128i *)v72 == v73 )
        goto LABEL_35;
    }
LABEL_8:
    while ( 2 )
    {
      v7 = (__int64)&v4[-2].m128i_i64[1];
      if ( !v4 )
        v7 = 0;
      v8 = sub_AA4FF0(v7);
      if ( !v8 )
        BUG();
      v9 = *(unsigned __int8 *)(v8 - 24);
      if ( (_BYTE)v9 == 85 )
      {
        v10 = *(_QWORD *)(v8 - 56);
        if ( !v10 )
          goto LABEL_7;
        if ( *(_BYTE *)v10 )
          goto LABEL_7;
        v3 = *(__m128i **)(v8 + 56);
        if ( *(__m128i **)(v10 + 24) != v3 || (*(_BYTE *)(v10 + 33) & 0x20) == 0 || *(_DWORD *)(v10 + 36) != 86 )
          goto LABEL_7;
      }
      else
      {
        v5 = (unsigned int)(v9 - 39);
        if ( (unsigned int)v5 > 0x38 )
          goto LABEL_7;
        v6 = 0x100060000000001LL;
        if ( !_bittest64(&v6, v5) )
          goto LABEL_7;
      }
      v11 = 4LL * (*(_DWORD *)(v8 - 20) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v8 - 17) & 0x40) != 0 )
      {
        v12 = *(unsigned __int8 ***)(v8 - 32);
        v81 = &v12[v11];
      }
      else
      {
        v81 = (unsigned __int8 **)(v8 - 24);
        v12 = (unsigned __int8 **)(v8 - 24 - v11 * 8);
      }
      v13 = a1 + 32;
      if ( v12 != v81 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v14 = sub_BD3990(*v12, (__int64)v3);
            if ( *v14 != 3 )
              break;
            v3 = (__m128i *)&v85;
            v85 = (unsigned __int64)v14;
            sub_2DD9820(v13, (__int64 *)&v85, v15, v16, v17, v18);
LABEL_22:
            v12 += 4;
            if ( v81 == v12 )
              goto LABEL_33;
          }
          v85 = 0;
          v19 = sub_BD3990(*v12, (__int64)v3);
          if ( *v19 != 9 )
            goto LABEL_22;
          v20 = 4LL * (*((_DWORD *)v19 + 1) & 0x7FFFFFF);
          if ( (v19[7] & 0x40) != 0 )
          {
            v21 = (unsigned __int8 **)*((_QWORD *)v19 - 1);
            v22 = (unsigned __int8 *)&v21[v20];
          }
          else
          {
            v22 = v19;
            v21 = (unsigned __int8 **)&v19[-(v20 * 8)];
          }
          if ( v21 == (unsigned __int8 **)v22 )
            goto LABEL_22;
          v77 = v12;
          v23 = v21;
          do
          {
            v24 = sub_BD3990(*v23, (__int64)v3);
            if ( *v24 == 3 )
            {
              v3 = &v87;
              v87.m128i_i64[0] = (__int64)v24;
              sub_2DD9820(v13, v87.m128i_i64, v25, v26, v27, v28);
            }
            v23 += 4;
          }
          while ( v22 != (unsigned __int8 *)v23 );
          v12 = v77 + 4;
          if ( v81 == v77 + 4 )
          {
LABEL_33:
            v4 = (__m128i *)v4->m128i_i64[1];
            if ( v79 == v4 )
              goto LABEL_34;
            goto LABEL_8;
          }
        }
      }
LABEL_7:
      v4 = (__m128i *)v4->m128i_i64[1];
      if ( v79 == v4 )
        goto LABEL_34;
      continue;
    }
  }
LABEL_35:
  for ( i = *(_QWORD *)(a2 + 16); a2 + 8 != i; i = *(_QWORD *)(i + 8) )
  {
    v30 = 0;
    if ( i )
      v30 = i - 56;
    v31 = v30;
    if ( sub_B2FC80(v30) )
      continue;
    if ( (*(_BYTE *)(v31 + 33) & 0x1C) != 0 )
      continue;
    v85 = *(_QWORD *)(v31 + 72);
    if ( (unsigned __int8)sub_A73380((__int64 *)&v85, "bss-section", 0xBu) )
      continue;
    v84 = *(_QWORD *)(v31 + 72);
    if ( (unsigned __int8)sub_A73380(&v84, "data-section", 0xCu) )
      continue;
    v83 = *(_QWORD *)(v31 + 72);
    if ( (unsigned __int8)sub_A73380(&v83, "relro-section", 0xDu) )
      continue;
    v32 = "rodata-section";
    v87.m128i_i64[0] = *(_QWORD *)(v31 + 72);
    if ( (unsigned __int8)sub_A73380(v87.m128i_i64, "rodata-section", 0xEu) )
      continue;
    if ( *(_QWORD *)a1 )
    {
      v32 = (const char *)v31;
      if ( !(unsigned __int8)sub_23CF1F0(*(_DWORD **)a1, (_BYTE *)v31) )
        continue;
    }
    v34 = *(_BYTE *)(v31 + 32) & 0xF;
    if ( (!*(_BYTE *)(a1 + 19) || v34) && (unsigned int)v34 - 7 > 1 )
      continue;
    v82 = 0;
    v80 = 0;
    v78 = *(_DWORD *)(*(_QWORD *)(v31 + 8) + 8LL) >> 8;
    if ( (*(_BYTE *)(v31 + 35) & 4) != 0 )
    {
      v65 = sub_B31D10(v31, (__int64)v32, v33);
      v82 = v65;
      v80 = v66;
      if ( *(_BYTE *)(a1 + 24) )
      {
        if ( v66 > 0x10 )
        {
          if ( !(*(_QWORD *)v65 ^ 0x5F2C415441445F5FLL | *(_QWORD *)(v65 + 8) ^ 0x6E6972747366635FLL)
            && *(_BYTE *)(v65 + 16) == 103 )
          {
            continue;
          }
          if ( v66 <= 0x16 )
          {
            if ( v66 <= 0x14 )
              goto LABEL_51;
          }
          else if ( !(*(_QWORD *)v65 ^ 0x5F2C415441445F5FLL | *(_QWORD *)(v65 + 8) ^ 0x6C635F636A626F5FLL)
                 && *(_DWORD *)(v65 + 16) == 1920168801
                 && *(_WORD *)(v65 + 20) == 26213
                 && *(_BYTE *)(v65 + 22) == 115 )
          {
            continue;
          }
          if ( !(*(_QWORD *)v65 ^ 0x5F2C415441445F5FLL | *(_QWORD *)(v65 + 8) ^ 0x65735F636A626F5FLL)
            && *(_DWORD *)(v65 + 16) == 1717924460
            && *(_BYTE *)(v65 + 20) == 115 )
          {
            continue;
          }
        }
      }
    }
LABEL_51:
    v35 = sub_BD5D20(v31);
    if ( v36 > 4 && *(_DWORD *)v35 == 1836477548 && v35[4] == 46 )
      continue;
    v37 = sub_BD5D20(v31);
    if ( v38 > 5 && *(_DWORD *)v37 == 1986817070 && *((_WORD *)v37 + 2) == 11885 )
      continue;
    if ( !*(_DWORD *)(a1 + 48) )
    {
      v39 = *(_QWORD **)(a1 + 64);
      v40 = 8LL * *(unsigned int *)(a1 + 72);
      v41 = (__int64)&v39[(unsigned __int64)v40 / 8];
      v42 = v40 >> 3;
      v38 = v40 >> 5;
      if ( v38 )
      {
        v38 = (unsigned __int64)&v39[4 * v38];
        while ( v31 != *v39 )
        {
          if ( v31 == v39[1] )
          {
            ++v39;
            break;
          }
          if ( v31 == v39[2] )
          {
            v39 += 2;
            break;
          }
          if ( v31 == v39[3] )
          {
            v39 += 3;
            break;
          }
          v39 += 4;
          if ( (_QWORD *)v38 == v39 )
          {
            v42 = (v41 - (__int64)v39) >> 3;
            goto LABEL_143;
          }
        }
LABEL_63:
        if ( (_QWORD *)v41 == v39 )
          goto LABEL_64;
        continue;
      }
LABEL_143:
      if ( v42 != 2 )
      {
        if ( v42 != 3 )
        {
          if ( v42 != 1 )
            goto LABEL_64;
          goto LABEL_146;
        }
        if ( v31 == *v39 )
          goto LABEL_63;
        ++v39;
      }
      if ( v31 == *v39 )
        goto LABEL_63;
      ++v39;
LABEL_146:
      if ( v31 != *v39 )
        goto LABEL_64;
      goto LABEL_63;
    }
    v41 = *(unsigned int *)(a1 + 56);
    v67 = *(_QWORD *)(a1 + 40);
    if ( !(_DWORD)v41 )
      goto LABEL_64;
    v41 = (unsigned int)(v41 - 1);
    v68 = v41 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v69 = *(_QWORD *)(v67 + 8LL * v68);
    if ( v31 != v69 )
    {
      v38 = 1;
      while ( v69 != -4096 )
      {
        v68 = v41 & (v38 + v68);
        v69 = *(_QWORD *)(v67 + 8LL * v68);
        if ( v31 == v69 )
          goto LABEL_37;
        v38 = (unsigned int)(v38 + 1);
      }
LABEL_64:
      if ( (*(_BYTE *)(v31 + 34) & 1) == 0 || (*(_BYTE *)sub_B31490(v31, v41, v38) & 4) == 0 )
      {
        v74 = *(_QWORD *)(v31 + 24);
        v76 = sub_AE5020(v70, v74);
        v43.m128i_i64[0] = sub_9208B0(v70, v74);
        v87 = v43;
        v85 = ((1LL << v76) + ((unsigned __int64)(v43.m128i_i64[0] + 7) >> 3) - 1) >> v76 << v76;
        v86 = v43.m128i_i8[8];
        if ( sub_CA1930(&v85) < (unsigned __int64)*(unsigned int *)(a1 + 8)
          && sub_CA1930(&v85) >= (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          if ( *(_QWORD *)a1 && (unsigned __int8)(sub_31578C0(v31) - 15) <= 2u )
          {
            v44 = &v101;
            v87.m128i_i32[0] = v78;
            v87.m128i_i64[1] = v82;
            v88 = v80;
          }
          else
          {
            if ( (*(_BYTE *)(v31 + 80) & 1) != 0 )
              v44 = &v95;
            else
              v44 = &v89;
            v87.m128i_i32[0] = v78;
            v87.m128i_i64[1] = v82;
            v88 = v80;
          }
          v47 = sub_2DDA0E0((__int64)v44, &v87);
          v48 = *(unsigned int *)(v47 + 8);
          if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v47 + 12) )
          {
            sub_C8D5F0(v47, (const void *)(v47 + 16), v48 + 1, 8u, v45, v46);
            v48 = *(unsigned int *)(v47 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v47 + 8 * v48) = v31;
          ++*(_DWORD *)(v47 + 8);
        }
      }
    }
LABEL_37:
    ;
  }
  v49 = v93;
  v50 = &v93[10 * (unsigned int)v94];
  if ( v93 == v50 )
  {
    v51 = 0;
  }
  else
  {
    v51 = 0;
    do
    {
      if ( v49[8] > 1 )
        v51 |= sub_2DD7CD0(a1, (__int64)(v49 + 6), (_QWORD **)a2, 0, *v49);
      v49 += 10;
    }
    while ( v50 != v49 );
  }
  v52 = (unsigned __int64)v105;
  v53 = &v105[40 * (unsigned int)v106];
  if ( v105 == v53 )
  {
    if ( !*(_BYTE *)(a1 + 20) )
      goto LABEL_96;
    v54 = v99;
    v55 = &v99[10 * (unsigned int)v100];
    if ( v55 == v99 )
      goto LABEL_96;
  }
  else
  {
    do
    {
      if ( *(_DWORD *)(v52 + 32) > 1u )
        v51 |= sub_2DD7CD0(a1, v52 + 24, (_QWORD **)a2, 0, *(_DWORD *)v52);
      v52 += 40LL;
    }
    while ( v53 != (_BYTE *)v52 );
    if ( !*(_BYTE *)(a1 + 20) )
      goto LABEL_91;
    v54 = v99;
    v55 = &v99[10 * (unsigned int)v100];
    if ( v99 == v55 )
      goto LABEL_91;
  }
  do
  {
    if ( v54[8] > 1 )
      v51 |= sub_2DD7CD0(a1, (__int64)(v54 + 6), (_QWORD **)a2, 1, *v54);
    v54 += 10;
  }
  while ( v55 != v54 );
LABEL_91:
  v56 = v105;
  v52 = (unsigned __int64)&v105[40 * (unsigned int)v106];
  if ( v105 != (_BYTE *)v52 )
  {
    do
    {
      v52 -= 40LL;
      v57 = *(_QWORD *)(v52 + 24);
      if ( v57 != v52 + 40 )
        _libc_free(v57);
    }
    while ( v56 != (_BYTE *)v52 );
    v52 = (unsigned __int64)v105;
  }
LABEL_96:
  if ( (_BYTE *)v52 != v107 )
    _libc_free(v52);
  sub_C7D6A0(v102, 32LL * v104, 8);
  v58 = v99;
  v59 = &v99[10 * (unsigned int)v100];
  if ( v99 != v59 )
  {
    do
    {
      v59 -= 10;
      v60 = *((_QWORD *)v59 + 3);
      if ( (unsigned int *)v60 != v59 + 10 )
        _libc_free(v60);
    }
    while ( v58 != v59 );
    v59 = v99;
  }
  if ( v59 != (unsigned int *)&v101 )
    _libc_free((unsigned __int64)v59);
  sub_C7D6A0(v96, 32LL * v98, 8);
  v61 = v93;
  v62 = &v93[10 * (unsigned int)v94];
  if ( v93 != v62 )
  {
    do
    {
      v62 -= 10;
      v63 = *((_QWORD *)v62 + 3);
      if ( (unsigned int *)v63 != v62 + 10 )
        _libc_free(v63);
    }
    while ( v61 != v62 );
    v62 = v93;
  }
  if ( v62 != (unsigned int *)&v95 )
    _libc_free((unsigned __int64)v62);
  sub_C7D6A0(v90, 32LL * v92, 8);
  return v51;
}
