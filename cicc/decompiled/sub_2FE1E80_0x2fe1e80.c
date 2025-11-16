// Function: sub_2FE1E80
// Address: 0x2fe1e80
//
__int64 __fastcall sub_2FE1E80(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        unsigned int *a7,
        int a8,
        __int64 a9)
{
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 (*v15)(void); // rdx
  _QWORD *v16; // rax
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r15
  int v21; // esi
  __int64 v22; // r9
  unsigned __int32 v23; // eax
  unsigned __int32 v24; // ecx
  unsigned int v25; // esi
  int v26; // r14d
  __int64 v27; // r8
  unsigned int v28; // edi
  _DWORD *v29; // rdx
  int v30; // eax
  char v31; // r15
  unsigned int v32; // eax
  __int64 v33; // r13
  unsigned __int8 *v34; // rsi
  __int64 v35; // rax
  unsigned __int8 *v36; // rdx
  __int64 v37; // rcx
  _QWORD *v38; // rax
  __int64 v39; // r13
  __int64 v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // r15
  __int32 v43; // eax
  int v44; // eax
  const __m128i *v45; // rdx
  unsigned __int8 *v46; // rsi
  __int64 v47; // r15
  __int64 v48; // rax
  unsigned __int8 *v49; // rdx
  __int64 v50; // rcx
  _QWORD *v51; // rax
  __int64 v52; // rbx
  __int64 v53; // r15
  __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v56; // r14
  __int32 v57; // eax
  int v58; // eax
  const __m128i *v59; // rdx
  __int64 v60; // r8
  __int64 v61; // r9
  int v62; // edx
  void (*v63)(); // rax
  __int64 v64; // rax
  unsigned __int64 v65; // rcx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 result; // rax
  int v70; // eax
  unsigned __int32 v71; // ebx
  char v72; // al
  int v73; // eax
  int v74; // r11d
  _DWORD *v75; // rcx
  int v76; // edx
  int v77; // eax
  int v78; // edi
  __int64 v79; // r10
  unsigned int v80; // eax
  int v81; // esi
  int v82; // r9d
  _DWORD *v83; // r8
  int v84; // eax
  int v85; // r9d
  __int64 v86; // r8
  int v87; // esi
  unsigned int v88; // r14d
  _DWORD *v89; // rdi
  int v90; // eax
  unsigned __int64 v91; // [rsp+8h] [rbp-108h]
  int v93; // [rsp+18h] [rbp-F8h]
  __int8 v94; // [rsp+1Ch] [rbp-F4h]
  char v95; // [rsp+1Dh] [rbp-F3h]
  char v96; // [rsp+1Eh] [rbp-F2h]
  __int8 v97; // [rsp+1Eh] [rbp-F2h]
  char v98; // [rsp+1Fh] [rbp-F1h]
  __int64 v101; // [rsp+30h] [rbp-E0h]
  int v102; // [rsp+38h] [rbp-D8h]
  unsigned int v103; // [rsp+3Ch] [rbp-D4h]
  unsigned int v104; // [rsp+40h] [rbp-D0h]
  unsigned __int32 v105; // [rsp+48h] [rbp-C8h]
  __int64 v106; // [rsp+48h] [rbp-C8h]
  __int64 v107; // [rsp+48h] [rbp-C8h]
  __int64 v108; // [rsp+48h] [rbp-C8h]
  __int64 v109; // [rsp+48h] [rbp-C8h]
  __int64 v110; // [rsp+48h] [rbp-C8h]
  int v111; // [rsp+50h] [rbp-C0h]
  __int64 v112; // [rsp+50h] [rbp-C0h]
  unsigned int v113; // [rsp+58h] [rbp-B8h]
  int v114; // [rsp+5Ch] [rbp-B4h]
  __int64 v115; // [rsp+60h] [rbp-B0h]
  unsigned int v116; // [rsp+60h] [rbp-B0h]
  __int64 v117; // [rsp+68h] [rbp-A8h]
  unsigned int v118; // [rsp+68h] [rbp-A8h]
  char v119; // [rsp+68h] [rbp-A8h]
  unsigned __int8 *v122; // [rsp+88h] [rbp-88h] BYREF
  unsigned __int8 *v123; // [rsp+90h] [rbp-80h] BYREF
  __int64 v124; // [rsp+98h] [rbp-78h]
  __int64 v125; // [rsp+A0h] [rbp-70h]
  __m128i v126; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v127; // [rsp+C0h] [rbp-50h]
  __int64 v128; // [rsp+C8h] [rbp-48h]
  __int64 v129; // [rsp+D0h] [rbp-40h]

  v10 = sub_2E88D60(a2);
  v101 = 0;
  v11 = *(__int64 **)(v10 + 16);
  v12 = *(_QWORD *)(v10 + 32);
  v13 = v10;
  v14 = *v11;
  v15 = *(__int64 (**)(void))(*v11 + 128);
  if ( v15 != sub_2DAC790 )
  {
    v101 = v15();
    v14 = **(_QWORD **)(v13 + 16);
  }
  v16 = (_QWORD *)(*(__int64 (**)(void))(v14 + 200))();
  v17 = sub_2E8A250(a2, 0, v101, v16);
  v18 = *(_QWORD *)(a3 + 32);
  v19 = *(_QWORD *)(a2 + 32);
  v20 = v18 + 40LL * a7[3];
  v117 = v18 + 40LL * a7[1];
  v21 = *(_DWORD *)(v117 + 8);
  v102 = v21;
  v115 = v19 + 40LL * a7[4];
  v22 = *(unsigned int *)(v19 + 40LL * a7[2] + 8);
  v114 = *(_DWORD *)(v115 + 8);
  v111 = *(_DWORD *)(v20 + 8);
  v93 = *(_DWORD *)(v19 + 8);
  if ( v21 < 0 )
  {
    v104 = *(_DWORD *)(v19 + 40LL * a7[2] + 8);
    v110 = v17;
    sub_2EBE590(v12, v21, v17, 0);
    v22 = v104;
    v17 = v110;
  }
  if ( (int)v22 < 0 )
  {
    v109 = v17;
    sub_2EBE590(v12, v22, v17, 0);
    v17 = v109;
  }
  if ( v111 < 0 )
  {
    v108 = v17;
    sub_2EBE590(v12, v111, v17, 0);
    v17 = v108;
  }
  if ( v114 < 0 )
  {
    v107 = v17;
    sub_2EBE590(v12, v114, v17, 0);
    v17 = v107;
  }
  if ( v93 < 0 )
  {
    v106 = v17;
    sub_2EBE590(v12, v93, v17, 0);
    v17 = v106;
  }
  v23 = sub_2EC06C0(v12, v17, byte_3F871B3, 0, v17, v22);
  v105 = v23;
  v24 = v23;
  v25 = *(_DWORD *)(a9 + 24);
  if ( !v25 )
  {
    ++*(_QWORD *)a9;
    goto LABEL_104;
  }
  v26 = 37 * v23;
  v27 = *(_QWORD *)(a9 + 8);
  v28 = (v25 - 1) & (37 * v23);
  v29 = (_DWORD *)(v27 + 8LL * v28);
  v30 = *v29;
  if ( v24 == *v29 )
    goto LABEL_15;
  v74 = 1;
  v75 = 0;
  while ( v30 != -1 )
  {
    if ( v75 || v30 != -2 )
      v29 = v75;
    v28 = (v25 - 1) & (v74 + v28);
    v30 = *(_DWORD *)(v27 + 8LL * v28);
    if ( v105 == v30 )
      goto LABEL_15;
    ++v74;
    v75 = v29;
    v29 = (_DWORD *)(v27 + 8LL * v28);
  }
  if ( !v75 )
    v75 = v29;
  ++*(_QWORD *)a9;
  v76 = *(_DWORD *)(a9 + 16) + 1;
  if ( 4 * v76 >= 3 * v25 )
  {
LABEL_104:
    sub_A09770(a9, 2 * v25);
    v77 = *(_DWORD *)(a9 + 24);
    if ( v77 )
    {
      v78 = v77 - 1;
      v79 = *(_QWORD *)(a9 + 8);
      v80 = (v77 - 1) & (37 * v105);
      v75 = (_DWORD *)(v79 + 8LL * v80);
      v81 = *v75;
      v76 = *(_DWORD *)(a9 + 16) + 1;
      if ( v105 != *v75 )
      {
        v82 = 1;
        v83 = 0;
        while ( v81 != -1 )
        {
          if ( v81 == -2 && !v83 )
            v83 = v75;
          v80 = v78 & (v82 + v80);
          v75 = (_DWORD *)(v79 + 8LL * v80);
          v81 = *v75;
          if ( v105 == *v75 )
            goto LABEL_100;
          ++v82;
        }
        if ( v83 )
          v75 = v83;
      }
      goto LABEL_100;
    }
    goto LABEL_133;
  }
  if ( v25 - *(_DWORD *)(a9 + 20) - v76 <= v25 >> 3 )
  {
    sub_A09770(a9, v25);
    v84 = *(_DWORD *)(a9 + 24);
    if ( v84 )
    {
      v85 = 1;
      v86 = *(_QWORD *)(a9 + 8);
      v87 = v84 - 1;
      v88 = (v84 - 1) & v26;
      v76 = *(_DWORD *)(a9 + 16) + 1;
      v89 = 0;
      v75 = (_DWORD *)(v86 + 8LL * v88);
      v90 = *v75;
      if ( v105 != *v75 )
      {
        while ( v90 != -1 )
        {
          if ( v90 == -2 && !v89 )
            v89 = v75;
          v88 = v87 & (v85 + v88);
          v75 = (_DWORD *)(v86 + 8LL * v88);
          v90 = *v75;
          if ( v105 == *v75 )
            goto LABEL_100;
          ++v85;
        }
        if ( v89 )
          v75 = v89;
      }
      goto LABEL_100;
    }
LABEL_133:
    ++*(_DWORD *)(a9 + 16);
    BUG();
  }
LABEL_100:
  *(_DWORD *)(a9 + 16) = v76;
  if ( *v75 != -1 )
    --*(_DWORD *)(a9 + 20);
  *(_QWORD *)v75 = v105;
LABEL_15:
  v91 = sub_2FE06E0(a1, a4, a2, a3);
  v95 = ((*(_BYTE *)(v117 + 3) & 0x40) != 0) & ((*(_BYTE *)(v117 + 3) >> 4) ^ 1);
  v31 = ((*(_BYTE *)(v20 + 3) >> 4) ^ 1) & ((*(_BYTE *)(v20 + 3) & 0x40) != 0);
  v96 = ((*(_BYTE *)(v115 + 3) & 0x40) != 0) & ((*(_BYTE *)(v115 + 3) >> 4) ^ 1);
  if ( a4 == 2 )
  {
    v98 = 1;
    v118 = a7[3];
    v113 = a7[1];
    v116 = a7[2];
    v103 = a7[4];
    v72 = v96;
    v96 = v31;
    v31 = v72;
    v73 = v111;
    v111 = v114;
    v114 = v73;
  }
  else
  {
    if ( a4 > 2 )
    {
      if ( a4 != 3 )
        BUG();
      v98 = 1;
      v118 = a7[3];
      v32 = a7[1];
      goto LABEL_19;
    }
    if ( a4 )
    {
      v98 = 1;
      v118 = a7[1];
      v32 = a7[3];
LABEL_19:
      v113 = v32;
      v116 = a7[4];
      v103 = a7[2];
      goto LABEL_20;
    }
    v96 = v31;
    v98 = 0;
    v31 = ((*(_BYTE *)(v115 + 3) & 0x40) != 0) & ((*(_BYTE *)(v115 + 3) >> 4) ^ 1);
    v70 = v111;
    v111 = v114;
    v114 = v70;
    v118 = a7[1];
    v113 = a7[3];
    v116 = a7[2];
    v103 = a7[4];
  }
LABEL_20:
  v33 = *(_QWORD *)(v101 + 8) - 40 * HIDWORD(v91);
  v34 = *(unsigned __int8 **)(a3 + 56);
  v123 = v34;
  if ( !v34 )
  {
    v35 = *(_QWORD *)(a3 + 48);
    v36 = (unsigned __int8 *)(v35 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v35 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v124 = 0;
      v125 = 0;
      v122 = 0;
      v38 = sub_2E7B380((_QWORD *)v13, v33, &v122, 1u);
      goto LABEL_28;
    }
    goto LABEL_22;
  }
  sub_B96E90((__int64)&v123, (__int64)v34, 1);
  v35 = *(_QWORD *)(a3 + 48);
  v34 = v123;
  v36 = (unsigned __int8 *)(v35 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v35 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
LABEL_22:
    v37 = 0;
    if ( (v35 & 7) == 3 && v36[7] )
      v37 = *(_QWORD *)&v36[8 * v36[6] + 16 + 8 * *(int *)v36 + 8 * (__int64)(v36[5] + v36[4])];
    goto LABEL_25;
  }
  v37 = 0;
LABEL_25:
  v124 = v37;
  v125 = 0;
  v122 = v34;
  if ( v34 )
    sub_B96E90((__int64)&v122, (__int64)v34, 1);
  v38 = sub_2E7B380((_QWORD *)v13, v33, &v122, 1u);
LABEL_28:
  v39 = (__int64)v38;
  if ( v124 )
    sub_2E882B0((__int64)v38, v13, v124);
  v126.m128i_i64[0] = 0x10000000;
  v127 = 0;
  v126.m128i_i32[2] = v105;
  v128 = 0;
  v129 = 0;
  sub_2E8EAD0(v39, v13, &v126);
  if ( v122 )
    sub_B91220((__int64)&v122, (__int64)v122);
  if ( v123 )
    sub_B91220((__int64)&v123, (__int64)v123);
  v40 = *(_QWORD *)(a3 + 32);
  v41 = v40 + 40LL * (unsigned int)sub_2E88F80(a3);
  if ( v41 != *(_QWORD *)(a3 + 32) )
  {
    v94 = (v31 & 1) << 6;
    v97 = (v96 & 1) << 6;
    v42 = *(_QWORD *)(a3 + 32);
    while ( 1 )
    {
      v44 = sub_2EAB0A0(v42);
      if ( !v44 )
        goto LABEL_39;
      if ( v118 == v44 )
        break;
      if ( v113 == v44 )
      {
        v126.m128i_i64[0] = 0;
        v127 = 0;
        v126.m128i_i8[3] = v94;
        v43 = v111;
        v126.m128i_i16[1] &= 0xF00Fu;
LABEL_38:
        v126.m128i_i32[0] &= 0xFFF000FF;
        v126.m128i_i32[2] = v43;
        v128 = 0;
        v129 = 0;
        sub_2E8EAD0(v39, v13, &v126);
LABEL_39:
        v42 += 40;
        if ( v41 == v42 )
          goto LABEL_44;
      }
      else
      {
        v45 = (const __m128i *)v42;
        v42 += 40;
        sub_2E8EAD0(v39, v13, v45);
        if ( v41 == v42 )
          goto LABEL_44;
      }
    }
    v126.m128i_i64[0] = 0;
    v127 = 0;
    v126.m128i_i8[3] = v97;
    v43 = v114;
    v126.m128i_i16[1] &= 0xF00Fu;
    goto LABEL_38;
  }
LABEL_44:
  sub_2E8FE40(v39, v13, a3);
  if ( v98 )
  {
    v71 = v105;
    v105 = v102;
    v102 = v71;
    v119 = v95;
    v95 = v98;
  }
  else
  {
    v119 = 1;
  }
  v46 = *(unsigned __int8 **)(a2 + 56);
  v47 = *(_QWORD *)(v101 + 8) - 40LL * (unsigned int)v91;
  v123 = v46;
  if ( !v46 )
  {
    v48 = *(_QWORD *)(a2 + 48);
    v49 = (unsigned __int8 *)(v48 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v48 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v124 = 0;
      v125 = 0;
      v122 = 0;
      goto LABEL_53;
    }
    goto LABEL_48;
  }
  sub_B96E90((__int64)&v123, (__int64)v46, 1);
  v48 = *(_QWORD *)(a2 + 48);
  v46 = v123;
  v49 = (unsigned __int8 *)(v48 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v48 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
LABEL_48:
    v50 = 0;
    if ( (v48 & 7) == 3 && v49[7] )
      v50 = *(_QWORD *)&v49[8 * v49[6] + 16 + 8 * *(int *)v49 + 8 * (__int64)(v49[5] + v49[4])];
    goto LABEL_51;
  }
  v50 = 0;
LABEL_51:
  v124 = v50;
  v125 = 0;
  v122 = v46;
  if ( v46 )
    sub_B96E90((__int64)&v122, (__int64)v46, 1);
LABEL_53:
  v51 = sub_2E7B380((_QWORD *)v13, v47, &v122, 1u);
  v52 = (__int64)v51;
  if ( v124 )
    sub_2E882B0((__int64)v51, v13, v124);
  v126.m128i_i64[0] = 0x10000000;
  v127 = 0;
  v126.m128i_i32[2] = v93;
  v128 = 0;
  v129 = 0;
  sub_2E8EAD0(v52, v13, &v126);
  if ( v122 )
    sub_B91220((__int64)&v122, (__int64)v122);
  if ( v123 )
    sub_B91220((__int64)&v123, (__int64)v123);
  v53 = *(_QWORD *)(a2 + 32);
  v54 = v53 + 40LL * (unsigned int)sub_2E88F80(a2);
  if ( v54 != *(_QWORD *)(a2 + 32) )
  {
    v112 = v39;
    v55 = *(_QWORD *)(a2 + 32);
    v56 = v54;
    while ( 1 )
    {
      v58 = sub_2EAB0A0(v55);
      if ( !v58 )
        goto LABEL_64;
      if ( v116 == v58 )
        break;
      if ( v103 == v58 )
      {
        v126.m128i_i64[0] = 0;
        v127 = 0;
        *(__int32 *)((char *)v126.m128i_i32 + 3) = (v119 & 1) << 6;
        v57 = v105;
        v126.m128i_i16[1] &= 0xF00Fu;
LABEL_63:
        v126.m128i_i32[0] &= 0xFFF000FF;
        v126.m128i_i32[2] = v57;
        v128 = 0;
        v129 = 0;
        sub_2E8EAD0(v52, v13, &v126);
LABEL_64:
        v55 += 40;
        if ( v56 == v55 )
          goto LABEL_69;
      }
      else
      {
        v59 = (const __m128i *)v55;
        v55 += 40;
        sub_2E8EAD0(v52, v13, v59);
        if ( v56 == v55 )
        {
LABEL_69:
          v39 = v112;
          goto LABEL_70;
        }
      }
    }
    v126.m128i_i64[0] = 0;
    v127 = 0;
    *(__int32 *)((char *)v126.m128i_i32 + 3) = (v95 & 1) << 6;
    v57 = v102;
    v126.m128i_i16[1] &= 0xF00Fu;
    goto LABEL_63;
  }
LABEL_70:
  sub_2E8FE40(v52, v13, a2);
  v62 = *(_DWORD *)(a3 + 44) & *(_DWORD *)(a2 + 44) & 0xFFFFF3;
  *(_DWORD *)(v39 + 44) = *(_DWORD *)(a3 + 44) & *(_DWORD *)(a2 + 44) & 0xFFC7F3
                        | *(_DWORD *)(v39 + 44) & 0xC
                        | *(_DWORD *)(v39 + 44) & 0xFF000000;
  *(_DWORD *)(v52 + 44) = *(_DWORD *)(v52 + 44) & 0xC | v62 & 0xFFC7FF | *(_DWORD *)(v52 + 44) & 0xFF000000;
  v63 = *(void (**)())(*a1 + 704);
  if ( v63 != nullsub_1684 )
    ((void (__fastcall *)(__int64 *, __int64, __int64, __int64, __int64))v63)(a1, a2, a3, v39, v52);
  v64 = *(unsigned int *)(a5 + 8);
  if ( v64 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
  {
    sub_C8D5F0(a5, (const void *)(a5 + 16), v64 + 1, 8u, v60, v61);
    v64 = *(unsigned int *)(a5 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a5 + 8 * v64) = v39;
  v65 = *(unsigned int *)(a5 + 12);
  v66 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
  *(_DWORD *)(a5 + 8) = v66;
  if ( v66 + 1 > v65 )
  {
    sub_C8D5F0(a5, (const void *)(a5 + 16), v66 + 1, 8u, v60, v61);
    v66 = *(unsigned int *)(a5 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a5 + 8 * v66) = v52;
  ++*(_DWORD *)(a5 + 8);
  v67 = *(unsigned int *)(a6 + 8);
  if ( v67 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    sub_C8D5F0(a6, (const void *)(a6 + 16), v67 + 1, 8u, v60, v61);
    v67 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v67) = a3;
  v68 = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
  *(_DWORD *)(a6 + 8) = v68;
  if ( v68 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    sub_C8D5F0(a6, (const void *)(a6 + 16), v68 + 1, 8u, v60, v61);
    v68 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v68) = a2;
  ++*(_DWORD *)(a6 + 8);
  result = *(unsigned int *)(a2 + 64);
  if ( (_DWORD)result )
    *(_DWORD *)(v52 + 64) = result;
  return result;
}
