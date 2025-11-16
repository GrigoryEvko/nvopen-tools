// Function: sub_1B3FB90
// Address: 0x1b3fb90
//
__int64 __fastcall sub_1B3FB90(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // rsi
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 *v13; // rax
  unsigned int v14; // eax
  __int64 v15; // r15
  _QWORD *v16; // rax
  __int64 v17; // r12
  char v18; // r13
  __int64 v19; // rbx
  int v20; // r8d
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rbx
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r14
  __int64 *v33; // rbx
  __int64 v34; // rdx
  __int64 *v35; // r15
  __int64 v36; // rcx
  _QWORD *v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // r13
  int v44; // eax
  __int64 v45; // rax
  int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // rax
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // rsi
  __int64 v53; // rsi
  __m128i *v54; // r12
  __int64 v55; // rbx
  __int64 v56; // rax
  __int64 v57; // rbx
  __int64 *v58; // r12
  __int64 *v59; // r13
  unsigned __int64 v60; // rdx
  unsigned int v61; // ebx
  __int64 v62; // rax
  char *v63; // rcx
  char *v64; // rdx
  char **v65; // r8
  unsigned int v66; // ecx
  __int64 *v67; // rdx
  __int64 v68; // r10
  int v69; // eax
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  __int64 v72; // rax
  unsigned __int32 v73; // edx
  __int64 *v74; // r9
  unsigned __int32 v75; // ecx
  unsigned int v76; // edi
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r15
  __int64 v80; // rax
  char v81; // di
  __int64 v82; // r14
  __int64 i; // rbx
  __int64 v84; // rdx
  char *v85; // r13
  char **v86; // r10
  unsigned int v87; // edx
  char **v88; // rax
  __int64 v89; // r11
  char *v90; // rdx
  char v91; // r8
  __int64 v92; // rax
  unsigned __int32 v93; // eax
  char **v94; // r9
  unsigned __int32 v95; // edx
  unsigned int v96; // edi
  int v97; // ecx
  __int64 v98; // rdx
  char *v99; // rdi
  char **v100; // rax
  int v101; // r8d
  int v102; // ecx
  __int64 v103; // rdx
  char *v104; // rdi
  int v105; // r8d
  int v106; // r11d
  int v107; // edi
  char **v108; // r10
  unsigned int v109; // eax
  __int64 *v110; // rdx
  int v111; // ecx
  int v112; // ecx
  char **v113; // r8
  unsigned int v114; // eax
  __int64 *v115; // r10
  int v116; // edx
  __int64 v117; // [rsp+8h] [rbp-198h]
  __int64 v119; // [rsp+28h] [rbp-178h]
  __int64 v120; // [rsp+28h] [rbp-178h]
  __int64 *v121; // [rsp+28h] [rbp-178h]
  int v123; // [rsp+38h] [rbp-168h]
  int v124; // [rsp+38h] [rbp-168h]
  __int64 v125; // [rsp+48h] [rbp-158h] BYREF
  __int64 *v126; // [rsp+50h] [rbp-150h] BYREF
  __int64 v127; // [rsp+58h] [rbp-148h]
  _BYTE v128[128]; // [rsp+60h] [rbp-140h] BYREF
  __m128i v129; // [rsp+E0h] [rbp-C0h] BYREF
  char *v130; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v131; // [rsp+F8h] [rbp-A8h]
  __int64 v132; // [rsp+100h] [rbp-A0h]
  char v133; // [rsp+170h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a2 + 48);
  v126 = (__int64 *)v128;
  v127 = 0x800000000LL;
  if ( !v2 )
    BUG();
  if ( *(_BYTE *)(v2 - 8) == 77 )
  {
    v119 = v2 - 24;
    if ( (*(_DWORD *)(v2 - 4) & 0xFFFFFFF) != 0 )
    {
      v123 = *(_DWORD *)(v2 - 4) & 0xFFFFFFF;
      v3 = 0;
      v4 = v2;
      v5 = 0;
      do
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v4 - 1) & 0x40) != 0 )
            v6 = *(_QWORD *)(v4 - 32);
          else
            v6 = v119 - 24LL * (*(_DWORD *)(v4 - 4) & 0xFFFFFFF);
          v7 = *(_QWORD *)(v6 + 8 * v3 + 24LL * *(unsigned int *)(v4 + 32) + 8);
          v8 = v7;
          v11 = sub_1B3FB80((__int64 *)a1, v7);
          v12 = (unsigned int)v127;
          if ( (unsigned int)v127 >= HIDWORD(v127) )
          {
            v8 = (__int64)v128;
            v117 = v11;
            sub_16CD150((__int64)&v126, v128, 0, 16, v9, v10);
            v12 = (unsigned int)v127;
            v11 = v117;
          }
          v13 = &v126[2 * v12];
          *v13 = v7;
          v13[1] = v11;
          v14 = v127 + 1;
          LODWORD(v127) = v127 + 1;
          if ( v3 )
            break;
          v5 = v11;
          v3 = 1;
          if ( v123 == 1 )
            goto LABEL_35;
        }
        if ( v11 != v5 )
          v5 = 0;
        ++v3;
      }
      while ( v123 != (_DWORD)v3 );
LABEL_35:
      v24 = v14;
      v17 = v5;
      goto LABEL_26;
    }
LABEL_32:
    v17 = sub_1599EF0(*(__int64 ***)(a1 + 8));
    goto LABEL_28;
  }
  v15 = *(_QWORD *)(a2 + 8);
  if ( !v15 )
    goto LABEL_32;
  while ( 1 )
  {
    v16 = sub_1648700(v15);
    if ( (unsigned __int8)(*((_BYTE *)v16 + 16) - 25) <= 9u )
      break;
    v15 = *(_QWORD *)(v15 + 8);
    if ( !v15 )
      goto LABEL_32;
  }
  v17 = 0;
  v18 = 1;
  while ( 1 )
  {
    v19 = v16[5];
    v8 = v19;
    v21 = sub_1B3FB80((__int64 *)a1, v19);
    v22 = (unsigned int)v127;
    if ( (unsigned int)v127 >= HIDWORD(v127) )
    {
      v8 = (__int64)v128;
      v120 = v21;
      sub_16CD150((__int64)&v126, v128, 0, 16, v20, v21);
      v22 = (unsigned int)v127;
      v21 = v120;
    }
    v23 = &v126[2 * v22];
    *v23 = v19;
    v23[1] = v21;
    v24 = (unsigned int)(v127 + 1);
    LODWORD(v127) = v127 + 1;
    if ( v18 )
    {
      v17 = v21;
    }
    else if ( v21 != v17 )
    {
      v17 = 0;
    }
    v15 = *(_QWORD *)(v15 + 8);
    if ( !v15 )
      break;
    v18 = 0;
    while ( 1 )
    {
      v16 = sub_1648700(v15);
      v8 = *((unsigned __int8 *)v16 + 16);
      if ( (unsigned __int8)(v8 - 25) <= 9u )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        goto LABEL_26;
    }
  }
LABEL_26:
  if ( !(_DWORD)v24 )
    goto LABEL_32;
  if ( v17 )
    goto LABEL_28;
  v26 = *(_QWORD *)(a2 + 48);
  if ( !v26 )
    BUG();
  if ( *(_BYTE *)(v26 - 8) != 77 )
    goto LABEL_38;
  v57 = 16 * v24;
  v58 = v126;
  v129.m128i_i64[0] = 0;
  v59 = &v126[(unsigned __int64)v57 / 8];
  v60 = ((((((v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 2) | (v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 4)
       | (((v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 2)
       | (v57 >> 4)
       | ((unsigned __int64)v57 >> 5)) >> 8;
  v61 = (((v60
         | (((((v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 2) | (v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 4)
         | (((v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 2)
         | (v57 >> 4)
         | ((unsigned __int64)v57 >> 5)) >> 16)
       | v60
       | (((((v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 2) | (v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 4)
       | (((v57 >> 4) | ((unsigned __int64)v57 >> 5)) >> 2)
       | (v57 >> 4)
       | ((unsigned __int64)v57 >> 5))
      + 1;
  if ( v61 > 8 )
  {
    v129.m128i_i8[8] &= ~1u;
    v72 = sub_22077B0(16LL * v61);
    LODWORD(v131) = v61;
    v130 = (char *)v72;
  }
  else
  {
    v129.m128i_i8[8] |= 1u;
  }
  v62 = v129.m128i_i8[8] & 1;
  v129.m128i_i64[1] = v62;
  if ( (_BYTE)v62 )
  {
    v63 = &v133;
    v64 = (char *)&v130;
  }
  else
  {
    v64 = v130;
    v63 = &v130[16 * (unsigned int)v131];
    if ( v130 == v63 )
      goto LABEL_88;
  }
  do
  {
    if ( v64 )
      *(_QWORD *)v64 = -8;
    v64 += 16;
  }
  while ( v64 != v63 );
  for ( LOBYTE(v62) = v129.m128i_i8[8]; ; LOBYTE(v62) = v129.m128i_i8[8] )
  {
LABEL_88:
    v69 = v62 & 1;
    if ( v69 )
    {
      v8 = 7;
      v65 = &v130;
    }
    else
    {
      v8 = (unsigned int)v131;
      v65 = (char **)v130;
      if ( !(_DWORD)v131 )
      {
        v73 = v129.m128i_u32[2];
        ++v129.m128i_i64[0];
        v74 = 0;
        v75 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
LABEL_102:
        v76 = 3 * v8;
        goto LABEL_103;
      }
      v8 = (unsigned int)(v131 - 1);
    }
    v66 = v8 & (((unsigned int)*v58 >> 9) ^ ((unsigned int)*v58 >> 4));
    v67 = (__int64 *)&v65[2 * v66];
    v68 = *v67;
    if ( *v67 == *v58 )
      goto LABEL_86;
    v74 = 0;
    v106 = 1;
    while ( v68 != -8 )
    {
      if ( v74 || v68 != -16 )
        v67 = v74;
      v66 = v8 & (v106 + v66);
      v68 = (__int64)v65[2 * v66];
      if ( *v58 == v68 )
        goto LABEL_86;
      ++v106;
      v74 = v67;
      v67 = (__int64 *)&v65[2 * v66];
    }
    if ( !v74 )
      v74 = v67;
    v73 = v129.m128i_u32[2];
    ++v129.m128i_i64[0];
    v75 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
    if ( !(_BYTE)v69 )
    {
      v8 = (unsigned int)v131;
      goto LABEL_102;
    }
    v76 = 24;
    v8 = 8;
LABEL_103:
    if ( 4 * v75 < v76 )
    {
      if ( (_DWORD)v8 - v129.m128i_i32[3] - v75 > (unsigned int)v8 >> 3 )
        goto LABEL_105;
      sub_1B3C040((__int64)&v129, v8);
      if ( (v129.m128i_i8[8] & 1) != 0 )
      {
        v112 = 7;
        v113 = &v130;
      }
      else
      {
        v113 = (char **)v130;
        if ( !(_DWORD)v131 )
        {
LABEL_229:
          v129.m128i_i32[2] = (2 * ((unsigned __int32)v129.m128i_i32[2] >> 1) + 2) | v129.m128i_i8[8] & 1;
          BUG();
        }
        v112 = v131 - 1;
      }
      v73 = v129.m128i_u32[2];
      v114 = v112 & (((unsigned int)*v58 >> 9) ^ ((unsigned int)*v58 >> 4));
      v74 = (__int64 *)&v113[2 * v114];
      v8 = *v74;
      if ( *v58 == *v74 )
        goto LABEL_105;
      v115 = 0;
      v116 = 1;
      while ( v8 != -8 )
      {
        if ( v8 == -16 && !v115 )
          v115 = v74;
        v114 = v112 & (v116 + v114);
        v74 = (__int64 *)&v113[2 * v114];
        v8 = *v74;
        if ( *v58 == *v74 )
          goto LABEL_196;
        ++v116;
      }
      if ( v115 )
        v74 = v115;
LABEL_196:
      v73 = v129.m128i_u32[2];
      goto LABEL_105;
    }
    sub_1B3C040((__int64)&v129, 2 * v8);
    if ( (v129.m128i_i8[8] & 1) != 0 )
    {
      v107 = 7;
      v108 = &v130;
    }
    else
    {
      v108 = (char **)v130;
      if ( !(_DWORD)v131 )
        goto LABEL_229;
      v107 = v131 - 1;
    }
    v73 = v129.m128i_u32[2];
    v109 = v107 & (((unsigned int)*v58 >> 9) ^ ((unsigned int)*v58 >> 4));
    v74 = (__int64 *)&v108[2 * v109];
    v8 = *v74;
    if ( *v74 == *v58 )
      goto LABEL_105;
    v110 = 0;
    v111 = 1;
    while ( v8 != -8 )
    {
      if ( !v110 && v8 == -16 )
        v110 = v74;
      v109 = v107 & (v111 + v109);
      v74 = (__int64 *)&v108[2 * v109];
      v8 = *v74;
      if ( *v58 == *v74 )
        goto LABEL_196;
      ++v111;
    }
    if ( !v110 )
      goto LABEL_196;
    v74 = v110;
    v73 = v129.m128i_u32[2];
LABEL_105:
    v129.m128i_i32[2] = (2 * (v73 >> 1) + 2) | v73 & 1;
    if ( *v74 != -8 )
      --v129.m128i_i32[3];
    *v74 = *v58;
    v74[1] = v58[1];
LABEL_86:
    v58 += 2;
    if ( v59 == v58 )
      break;
  }
  v77 = sub_157F280(a2);
  v79 = v78;
  v17 = v77;
  if ( v77 == v78 )
  {
LABEL_129:
    if ( (v129.m128i_i8[8] & 1) == 0 )
      j___libc_free_0(v130);
    LODWORD(v24) = v127;
    v26 = *(_QWORD *)(a2 + 48);
    if ( !v26 )
    {
LABEL_39:
      LOWORD(v130) = 260;
      v27 = *(_QWORD *)(a1 + 8);
      v129.m128i_i64[0] = a1 + 16;
      v28 = sub_1648B60(64);
      v32 = v28;
      if ( v28 )
      {
        sub_15F1EA0(v28, v27, 53, 0, 0, v26);
        *(_DWORD *)(v32 + 56) = v24;
        sub_164B780(v32, v129.m128i_i64);
        v8 = *(unsigned int *)(v32 + 56);
        sub_1648880(v32, v8, 1);
      }
      v33 = v126;
      v34 = (__int64)&v126[2 * (unsigned int)v127];
      if ( v126 != (__int64 *)v34 )
      {
        v35 = &v126[2 * (unsigned int)v127];
        do
        {
          v42 = *v33;
          v43 = v33[1];
          v44 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
          if ( v44 == *(_DWORD *)(v32 + 56) )
          {
            sub_15F55D0(v32, v8, v34, v29, v30, v31);
            v44 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
          }
          v45 = (v44 + 1) & 0xFFFFFFF;
          v8 = (unsigned int)(v45 - 1);
          v46 = v45 | *(_DWORD *)(v32 + 20) & 0xF0000000;
          *(_DWORD *)(v32 + 20) = v46;
          if ( (v46 & 0x40000000) != 0 )
            v36 = *(_QWORD *)(v32 - 8);
          else
            v36 = v32 - 24 * v45;
          v37 = (_QWORD *)(v36 + 24LL * (unsigned int)v8);
          if ( *v37 )
          {
            v8 = v37[1];
            v38 = v37[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v38 = v8;
            if ( v8 )
              *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v38;
          }
          *v37 = v43;
          if ( v43 )
          {
            v39 = *(_QWORD *)(v43 + 8);
            v37[1] = v39;
            if ( v39 )
            {
              v30 = (__int64)(v37 + 1);
              v8 = (unsigned __int64)(v37 + 1) | *(_QWORD *)(v39 + 16) & 3LL;
              *(_QWORD *)(v39 + 16) = v8;
            }
            v37[2] = (v43 + 8) | v37[2] & 3LL;
            *(_QWORD *)(v43 + 8) = v37;
          }
          v40 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
          v41 = (unsigned int)(v40 - 1);
          if ( (*(_BYTE *)(v32 + 23) & 0x40) != 0 )
            v29 = *(_QWORD *)(v32 - 8);
          else
            v29 = v32 - 24 * v40;
          v33 += 2;
          v34 = 3LL * *(unsigned int *)(v32 + 56);
          *(_QWORD *)(v29 + 8 * v41 + 24LL * *(unsigned int *)(v32 + 56) + 8) = v42;
        }
        while ( v35 != v33 );
      }
      v47 = sub_157EB90(a2);
      v129 = (__m128i)(unsigned __int64)sub_1632FA0(v47);
      v130 = 0;
      v131 = 0;
      v132 = 0;
      v17 = sub_13E3350(v32, &v129, 0, 1, v48);
      if ( v17 )
      {
        sub_15F20C0((_QWORD *)v32);
        goto LABEL_28;
      }
      v125 = 0;
      v49 = sub_157ED20(a2);
      v52 = 0;
      if ( v49 && &v125 != (__int64 *)(v49 + 48) )
      {
        v53 = *(_QWORD *)(v49 + 48);
        v125 = v53;
        if ( !v53 )
        {
          v129.m128i_i64[0] = 0;
          v54 = (__m128i *)(v32 + 48);
          goto LABEL_92;
        }
        sub_1623A60((__int64)&v125, v53, 2);
        v52 = v125;
      }
      v129.m128i_i64[0] = v52;
      v54 = (__m128i *)(v32 + 48);
      if ( v52 )
      {
        sub_1623A60((__int64)&v129, v52, 2);
        if ( v54 == &v129 )
        {
          if ( v129.m128i_i64[0] )
            sub_161E7C0((__int64)&v129, v129.m128i_i64[0]);
          goto LABEL_69;
        }
        v70 = *(_QWORD *)(v32 + 48);
        if ( !v70 )
        {
LABEL_95:
          v71 = (unsigned __int8 *)v129.m128i_i64[0];
          *(_QWORD *)(v32 + 48) = v129.m128i_i64[0];
          if ( v71 )
            sub_1623210((__int64)&v129, v71, (__int64)v54);
          goto LABEL_69;
        }
LABEL_94:
        sub_161E7C0((__int64)v54, v70);
        goto LABEL_95;
      }
LABEL_92:
      if ( v54 != &v129 )
      {
        v70 = *(_QWORD *)(v32 + 48);
        if ( v70 )
          goto LABEL_94;
        *(_QWORD *)(v32 + 48) = v129.m128i_i64[0];
      }
LABEL_69:
      v55 = *(_QWORD *)(a1 + 48);
      if ( v55 )
      {
        v56 = *(unsigned int *)(v55 + 8);
        if ( (unsigned int)v56 >= *(_DWORD *)(v55 + 12) )
        {
          sub_16CD150(*(_QWORD *)(a1 + 48), (const void *)(v55 + 16), 0, 8, v50, v51);
          v56 = *(unsigned int *)(v55 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v55 + 8 * v56) = v32;
        ++*(_DWORD *)(v55 + 8);
      }
      if ( v125 )
        sub_161E7C0((__int64)&v125, v125);
      v17 = v32;
      goto LABEL_28;
    }
LABEL_38:
    v26 -= 24;
    goto LABEL_39;
  }
  while ( 1 )
  {
    v80 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
    if ( (_DWORD)v80 == (unsigned __int32)v129.m128i_i32[2] >> 1 )
      break;
LABEL_125:
    v92 = *(_QWORD *)(v17 + 32);
    if ( !v92 )
      BUG();
    v17 = 0;
    if ( *(_BYTE *)(v92 - 8) == 77 )
      v17 = v92 - 24;
    if ( v79 == v17 )
      goto LABEL_129;
  }
  if ( (_DWORD)v80 )
  {
    v81 = *(_BYTE *)(v17 + 23);
    v82 = 8 * v80;
    for ( i = 0; v82 != i; i += 8 )
    {
      v91 = v81 & 0x40;
      if ( (v81 & 0x40) != 0 )
        v84 = *(_QWORD *)(v17 - 8);
      else
        v84 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
      v85 = *(char **)(i + v84 + 24LL * *(unsigned int *)(v17 + 56) + 8);
      if ( (v129.m128i_i8[8] & 1) != 0 )
      {
        v8 = 7;
        v86 = &v130;
      }
      else
      {
        v8 = (unsigned int)v131;
        v86 = (char **)v130;
        if ( !(_DWORD)v131 )
        {
          v93 = v129.m128i_u32[2];
          ++v129.m128i_i64[0];
          v94 = 0;
          v95 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
LABEL_134:
          v96 = 3 * v8;
          goto LABEL_135;
        }
        v8 = (unsigned int)(v131 - 1);
      }
      v87 = v8 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
      v88 = &v86[2 * v87];
      v89 = (__int64)*v88;
      if ( *v88 == v85 )
      {
        v90 = v88[1];
        goto LABEL_117;
      }
      v124 = 1;
      v94 = 0;
      while ( v89 != -8 )
      {
        if ( v89 != -16 || v94 )
          v88 = v94;
        v87 = v8 & (v124 + v87);
        v121 = (__int64 *)&v86[2 * v87];
        v89 = *v121;
        if ( v85 == (char *)*v121 )
        {
          v90 = (char *)v121[1];
          goto LABEL_117;
        }
        ++v124;
        v94 = v88;
        v88 = &v86[2 * v87];
      }
      v96 = 24;
      v8 = 8;
      if ( !v94 )
        v94 = v88;
      v93 = v129.m128i_u32[2];
      ++v129.m128i_i64[0];
      v95 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
      if ( (v129.m128i_i8[8] & 1) == 0 )
      {
        v8 = (unsigned int)v131;
        goto LABEL_134;
      }
LABEL_135:
      if ( v96 <= 4 * v95 )
      {
        sub_1B3C040((__int64)&v129, 2 * v8);
        if ( (v129.m128i_i8[8] & 1) != 0 )
        {
          v97 = 7;
          v8 = (__int64)&v130;
        }
        else
        {
          v8 = (__int64)v130;
          if ( !(_DWORD)v131 )
            goto LABEL_227;
          v97 = v131 - 1;
        }
        v93 = v129.m128i_u32[2];
        LODWORD(v98) = v97 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
        v94 = (char **)(v8 + 16LL * (unsigned int)v98);
        v99 = *v94;
        if ( *v94 != v85 )
        {
          v100 = 0;
          v101 = 1;
          while ( v99 != (char *)-8LL )
          {
            if ( v99 == (char *)-16LL && !v100 )
              v100 = v94;
            v98 = v97 & (unsigned int)(v98 + v101);
            v94 = (char **)(v8 + 16 * v98);
            v99 = *v94;
            if ( v85 == *v94 )
              goto LABEL_155;
            ++v101;
          }
          goto LABEL_153;
        }
      }
      else if ( (_DWORD)v8 - v129.m128i_i32[3] - v95 <= (unsigned int)v8 >> 3 )
      {
        sub_1B3C040((__int64)&v129, v8);
        if ( (v129.m128i_i8[8] & 1) != 0 )
        {
          v102 = 7;
          v8 = (__int64)&v130;
        }
        else
        {
          v8 = (__int64)v130;
          if ( !(_DWORD)v131 )
          {
LABEL_227:
            v129.m128i_i32[2] = (2 * ((unsigned __int32)v129.m128i_i32[2] >> 1) + 2) | v129.m128i_i8[8] & 1;
            BUG();
          }
          v102 = v131 - 1;
        }
        v93 = v129.m128i_u32[2];
        LODWORD(v103) = v102 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
        v94 = (char **)(v8 + 16LL * (unsigned int)v103);
        v104 = *v94;
        if ( *v94 != v85 )
        {
          v100 = 0;
          v105 = 1;
          while ( v104 != (char *)-8LL )
          {
            if ( v104 == (char *)-16LL && !v100 )
              v100 = v94;
            v103 = v102 & (unsigned int)(v103 + v105);
            v94 = (char **)(v8 + 16 * v103);
            v104 = *v94;
            if ( v85 == *v94 )
              goto LABEL_155;
            ++v105;
          }
LABEL_153:
          if ( v100 )
            v94 = v100;
LABEL_155:
          v93 = v129.m128i_u32[2];
        }
      }
      v129.m128i_i32[2] = (2 * (v93 >> 1) + 2) | v93 & 1;
      if ( *v94 != (char *)-8LL )
        --v129.m128i_i32[3];
      *v94 = v85;
      v90 = 0;
      v94[1] = 0;
      v81 = *(_BYTE *)(v17 + 23);
      v91 = v81 & 0x40;
LABEL_117:
      if ( v91 )
      {
        if ( v90 != *(char **)(*(_QWORD *)(v17 - 8) + 3 * i) )
          goto LABEL_125;
      }
      else if ( v90 != *(char **)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF) + 3 * i) )
      {
        goto LABEL_125;
      }
    }
  }
  if ( (v129.m128i_i8[8] & 1) == 0 )
    j___libc_free_0(v130);
LABEL_28:
  if ( v126 != (__int64 *)v128 )
    _libc_free((unsigned __int64)v126);
  return v17;
}
