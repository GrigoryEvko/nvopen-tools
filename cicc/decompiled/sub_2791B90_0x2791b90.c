// Function: sub_2791B90
// Address: 0x2791b90
//
void __fastcall sub_2791B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 **v7; // rdx
  unsigned __int8 *v8; // r13
  const char *v9; // rax
  unsigned __int16 v10; // r12
  unsigned __int64 v11; // rax
  char v12; // r8
  __int16 v13; // r12
  __int64 v14; // rdx
  char v15; // r9
  unsigned __int64 v16; // rax
  int v17; // ecx
  unsigned __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rbx
  unsigned __int8 *v21; // rsi
  __int64 *v22; // r12
  _QWORD *v23; // rdi
  __int64 *v24; // rax
  __int64 *v25; // rdi
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rsi
  int v33; // edx
  __int64 v34; // rdi
  int v35; // edx
  unsigned int v36; // ecx
  __int64 *v37; // rax
  __int64 v38; // r10
  void *v39; // rsi
  unsigned int v40; // ecx
  void **v41; // rax
  _QWORD *v42; // r10
  void *v43; // rax
  const __m128i *v44; // r12
  __int64 v45; // rax
  __int64 *v46; // rdx
  unsigned __int64 v47; // r8
  __m128i *v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rax
  unsigned int v51; // ecx
  __int64 v52; // rdx
  unsigned __int8 *v53; // r8
  __int64 v54; // rsi
  __int64 v55; // r12
  __int64 v56; // r12
  __int64 *v57; // rax
  __int64 v58; // rdx
  int v59; // eax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // r12
  unsigned __int8 v65; // al
  unsigned __int8 *v66; // rsi
  __int64 *v67; // r13
  __int64 v68; // rcx
  int v69; // edx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 *v73; // r14
  __int64 v74; // r12
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __m128i v80; // xmm2
  __m128i v81; // xmm3
  __m128i v82; // xmm4
  unsigned __int64 *v83; // r12
  unsigned __int64 *v84; // r13
  unsigned __int64 *v85; // r12
  unsigned __int64 v86; // rdi
  __int64 v87; // rsi
  unsigned __int8 *v88; // rsi
  const void *v89; // rsi
  char *v90; // r12
  __int64 v91; // rsi
  unsigned __int8 *v92; // rsi
  int v93; // eax
  int v94; // eax
  int v95; // edx
  int v96; // r9d
  unsigned __int64 *v97; // r13
  unsigned __int64 v98; // rdi
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned __int8 **v101; // [rsp+0h] [rbp-3E0h]
  char v103; // [rsp+14h] [rbp-3CCh]
  char v104; // [rsp+18h] [rbp-3C8h]
  char v105; // [rsp+1Ch] [rbp-3C4h]
  __int64 v106; // [rsp+20h] [rbp-3C0h]
  __int64 v108; // [rsp+30h] [rbp-3B0h]
  unsigned __int8 **v109; // [rsp+38h] [rbp-3A8h]
  __int64 v110; // [rsp+40h] [rbp-3A0h]
  void *v111; // [rsp+50h] [rbp-390h] BYREF
  __int64 v112; // [rsp+58h] [rbp-388h]
  const char *v113; // [rsp+60h] [rbp-380h]
  __m128i v114; // [rsp+68h] [rbp-378h]
  __int64 v115; // [rsp+78h] [rbp-368h]
  __m128i v116; // [rsp+80h] [rbp-360h]
  __m128i v117; // [rsp+90h] [rbp-350h]
  unsigned __int64 *v118; // [rsp+A0h] [rbp-340h] BYREF
  __int64 v119; // [rsp+A8h] [rbp-338h]
  _BYTE v120[320]; // [rsp+B0h] [rbp-330h] BYREF
  char v121; // [rsp+1F0h] [rbp-1F0h]
  int v122; // [rsp+1F4h] [rbp-1ECh]
  __int64 v123; // [rsp+1F8h] [rbp-1E8h]
  unsigned __int8 *v124; // [rsp+200h] [rbp-1E0h] BYREF
  __int64 v125; // [rsp+208h] [rbp-1D8h]
  const char *v126; // [rsp+210h] [rbp-1D0h]
  __m128i v127; // [rsp+218h] [rbp-1C8h] BYREF
  __int64 v128; // [rsp+228h] [rbp-1B8h]
  __m128i v129; // [rsp+230h] [rbp-1B0h] BYREF
  __m128i v130; // [rsp+240h] [rbp-1A0h] BYREF
  unsigned __int64 *v131; // [rsp+250h] [rbp-190h] BYREF
  unsigned int v132; // [rsp+258h] [rbp-188h]
  _BYTE v133[320]; // [rsp+260h] [rbp-180h] BYREF
  char v134; // [rsp+3A0h] [rbp-40h]
  int v135; // [rsp+3A4h] [rbp-3Ch]
  __int64 v136; // [rsp+3A8h] [rbp-38h]

  v7 = *(unsigned __int8 ***)(a4 + 32);
  v101 = &v7[2 * *(unsigned int *)(a4 + 40)];
  if ( v7 != v101 )
  {
    v109 = *(unsigned __int8 ***)(a4 + 32);
    while ( 1 )
    {
      v8 = *v109;
      v108 = (__int64)v109[1];
      v106 = *(_QWORD *)(a2 + 8);
      v9 = sub_BD5D20(a2);
      v10 = *(_WORD *)(a2 + 2);
      v127.m128i_i16[4] = 773;
      v124 = (unsigned __int8 *)v9;
      v126 = ".pre";
      v11 = 1LL << (v10 >> 1);
      v12 = v10 & 1;
      v13 = (v10 >> 7) & 7;
      _BitScanReverse64(&v11, v11);
      v125 = v14;
      v15 = 63 - (v11 ^ 0x3F);
      v16 = *((_QWORD *)v8 + 6) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int8 *)v16 == v8 + 48 )
      {
        v18 = 0;
      }
      else
      {
        if ( !v16 )
          BUG();
        v17 = *(unsigned __int8 *)(v16 - 24);
        v18 = v16 - 24;
        if ( (unsigned int)(v17 - 30) >= 0xB )
          v18 = 0;
      }
      v103 = v15;
      v104 = *(_BYTE *)(a2 + 72);
      v105 = v12;
      v110 = v18 + 24;
      v19 = sub_BD2C40(80, unk_3F10A14);
      v20 = (__int64)v19;
      if ( v19 )
        sub_B4D0A0((__int64)v19, v106, v108, (__int64)&v124, v105, v103, v13, v104, v110, 0);
      v21 = *(unsigned __int8 **)(a2 + 48);
      v22 = (__int64 *)(v20 + 48);
      v124 = v21;
      if ( !v21 )
        break;
      sub_B96E90((__int64)&v124, (__int64)v21, 1);
      if ( v22 == (__int64 *)&v124 )
      {
        if ( v124 )
          sub_B91220((__int64)&v124, (__int64)v124);
        goto LABEL_13;
      }
      v87 = *(_QWORD *)(v20 + 48);
      if ( v87 )
        goto LABEL_85;
LABEL_86:
      v88 = v124;
      *(_QWORD *)(v20 + 48) = v124;
      if ( v88 )
        sub_B976B0((__int64)&v124, v88, v20 + 48);
LABEL_13:
      v23 = *(_QWORD **)(a1 + 120);
      if ( v23 )
      {
        v24 = (__int64 *)sub_D694D0(v23, v20, 0, *(_QWORD *)(v20 + 40), 2u, 1u);
        v25 = *(__int64 **)(a1 + 120);
        if ( *(_BYTE *)v24 == 27 )
          sub_D75120(v25, v24, 1);
        else
          sub_D73680(v25, (__int64)v24, 1);
      }
      sub_B91FC0((__int64 *)&v111, a2);
      if ( v111 || v112 || v113 || v114.m128i_i64[0] )
        sub_B9A100(v20, (__int64 *)&v111);
      if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
      {
        v27 = sub_B91C10(a2, 6);
        if ( v27 )
          sub_B99FD0(v20, 6u, v27);
        if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
        {
          v28 = sub_B91C10(a2, 16);
          if ( v28 )
            sub_B99FD0(v20, 0x10u, v28);
          if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
          {
            v29 = sub_B91C10(a2, 4);
            if ( v29 )
              sub_B99FD0(v20, 4u, v29);
            if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
            {
              v30 = sub_B91C10(a2, 25);
              if ( v30 )
              {
                v31 = *(_QWORD *)(a1 + 112);
                v32 = *(_QWORD *)(a2 + 40);
                v33 = *(_DWORD *)(v31 + 24);
                v34 = *(_QWORD *)(v31 + 8);
                if ( !v33 )
                  goto LABEL_95;
                v35 = v33 - 1;
                v36 = v35 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                v37 = (__int64 *)(v34 + 16LL * v36);
                v38 = *v37;
                if ( v32 == *v37 )
                {
LABEL_31:
                  v39 = (void *)v37[1];
                }
                else
                {
                  v94 = 1;
                  while ( v38 != -4096 )
                  {
                    v26 = (unsigned int)(v94 + 1);
                    v36 = v35 & (v94 + v36);
                    v37 = (__int64 *)(v34 + 16LL * v36);
                    v38 = *v37;
                    if ( v32 == *v37 )
                      goto LABEL_31;
                    v94 = v26;
                  }
                  v39 = 0;
                }
                v40 = v35 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
                v41 = (void **)(v34 + 16LL * v40);
                v42 = *v41;
                if ( v8 == *v41 )
                {
LABEL_33:
                  v43 = v41[1];
                }
                else
                {
                  v93 = 1;
                  while ( v42 != (_QWORD *)-4096LL )
                  {
                    v26 = (unsigned int)(v93 + 1);
                    v40 = v35 & (v93 + v40);
                    v41 = (void **)(v34 + 16LL * v40);
                    v42 = *v41;
                    if ( v8 == *v41 )
                      goto LABEL_33;
                    v93 = v26;
                  }
                  v43 = 0;
                }
                if ( v43 == v39 )
LABEL_95:
                  sub_B99FD0(v20, 0x19u, v30);
              }
            }
          }
        }
      }
      v124 = v8;
      v125 = v20;
      v44 = (const __m128i *)&v124;
      v45 = *(unsigned int *)(a3 + 8);
      v46 = *(__int64 **)a3;
      v126 = 0;
      v127 = 0u;
      v47 = v45 + 1;
      if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v89 = (const void *)(a3 + 16);
        if ( v46 > (__int64 *)&v124 || &v124 >= (unsigned __int8 **)&v46[5 * v45] )
        {
          sub_C8D5F0(a3, v89, v47, 0x28u, v47, v26);
          v46 = *(__int64 **)a3;
          v45 = *(unsigned int *)(a3 + 8);
          v44 = (const __m128i *)&v124;
        }
        else
        {
          v90 = (char *)((char *)&v124 - (char *)v46);
          sub_C8D5F0(a3, v89, v47, 0x28u, v47, v26);
          v46 = *(__int64 **)a3;
          v45 = *(unsigned int *)(a3 + 8);
          v44 = (const __m128i *)&v90[*(_QWORD *)a3];
        }
      }
      v48 = (__m128i *)&v46[5 * v45];
      *v48 = _mm_loadu_si128(v44);
      v48[1] = _mm_loadu_si128(v44 + 1);
      v48[2].m128i_i64[0] = v44[2].m128i_i64[0];
      ++*(_DWORD *)(a3 + 8);
      sub_102B9D0(*(_QWORD *)(a1 + 16), v108);
      sub_102BD20(*(_QWORD *)(a1 + 16), v20);
      if ( a5 )
      {
        v49 = *(_QWORD *)(a5 + 8);
        v50 = *(unsigned int *)(a5 + 24);
        if ( (_DWORD)v50 )
        {
          v51 = (v50 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v52 = v49 + 16LL * v51;
          v53 = *(unsigned __int8 **)v52;
          if ( v8 == *(unsigned __int8 **)v52 )
          {
LABEL_39:
            if ( v52 != v49 + 16 * v50 )
            {
              v54 = *(_QWORD *)(a5 + 32);
              v55 = v54 + 16LL * *(unsigned int *)(v52 + 8);
              if ( v55 != v54 + 16LL * *(unsigned int *)(a5 + 40) )
              {
                sub_30EC360(*(_QWORD *)(a1 + 104), v20, v8);
                v56 = *(_QWORD *)(v55 + 8);
                sub_F57030((_BYTE *)v20, v56, 0);
                sub_BD84D0(v56, v20);
                v57 = *(__int64 **)a3;
                v58 = *(_QWORD *)a3 + 40LL * *(unsigned int *)(a3 + 8);
                if ( *(_QWORD *)a3 != v58 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      if ( v56 == v57[1] )
                        v57[1] = v20;
                      if ( *((_DWORD *)v57 + 4) == 4 )
                      {
                        if ( v56 == v57[3] )
                          v57[3] = v20;
                        if ( v56 == v57[4] )
                          break;
                      }
                      v57 += 5;
                      if ( (__int64 *)v58 == v57 )
                        goto LABEL_51;
                    }
                    v57[4] = v20;
                    v57 += 5;
                  }
                  while ( (__int64 *)v58 != v57 );
                }
LABEL_51:
                v59 = sub_278A710(a1 + 136, v56, 0);
                if ( v59 )
                  sub_27918D0(a1 + 352, v59, v56, *(_QWORD *)(v56 + 40));
                sub_278A7A0(a1 + 136, (_BYTE *)v56);
                sub_278C2C0((_QWORD *)a1, (_QWORD *)v56, v60, v61, v62, v63);
              }
            }
          }
          else
          {
            v95 = 1;
            while ( v53 != (unsigned __int8 *)-4096LL )
            {
              v96 = v95 + 1;
              v51 = (v50 - 1) & (v95 + v51);
              v52 = v49 + 16LL * v51;
              v53 = *(unsigned __int8 **)v52;
              if ( v8 == *(unsigned __int8 **)v52 )
                goto LABEL_39;
              v95 = v96;
            }
          }
        }
      }
      v109 += 2;
      if ( v101 == v109 )
        goto LABEL_55;
    }
    if ( v22 == (__int64 *)&v124 )
      goto LABEL_13;
    v87 = *(_QWORD *)(v20 + 48);
    if ( !v87 )
      goto LABEL_13;
LABEL_85:
    sub_B91220(v20 + 48, v87);
    goto LABEL_86;
  }
LABEL_55:
  v64 = sub_278B590(a2, (__int64 **)a3, a1);
  sub_30EC4B0(*(_QWORD *)(a1 + 104), a2);
  sub_BD84D0(a2, v64);
  v65 = *(_BYTE *)v64;
  if ( *(_BYTE *)v64 == 84 )
  {
    sub_102BD20(*(_QWORD *)(a1 + 16), v64);
    sub_BD6B90((unsigned __int8 *)v64, (unsigned __int8 *)a2);
    v65 = *(_BYTE *)v64;
  }
  if ( v65 > 0x1Cu )
  {
    v66 = *(unsigned __int8 **)(a2 + 48);
    v67 = (__int64 *)(v64 + 48);
    v124 = v66;
    if ( v66 )
    {
      sub_B96E90((__int64)&v124, (__int64)v66, 1);
      if ( v67 == (__int64 *)&v124 )
      {
        if ( v124 )
          sub_B91220(v64 + 48, (__int64)v124);
        goto LABEL_62;
      }
      v91 = *(_QWORD *)(v64 + 48);
      if ( !v91 )
        goto LABEL_104;
    }
    else
    {
      if ( v67 == (__int64 *)&v124 )
        goto LABEL_62;
      v91 = *(_QWORD *)(v64 + 48);
      if ( !v91 )
        goto LABEL_62;
    }
    sub_B91220(v64 + 48, v91);
LABEL_104:
    v92 = v124;
    *(_QWORD *)(v64 + 48) = v124;
    if ( v92 )
      sub_B976B0((__int64)&v124, v92, v64 + 48);
  }
LABEL_62:
  v68 = *(_QWORD *)(v64 + 8);
  v69 = *(unsigned __int8 *)(v68 + 8);
  if ( (unsigned int)(v69 - 17) <= 1 )
    LOBYTE(v69) = *(_BYTE *)(**(_QWORD **)(v68 + 16) + 8LL);
  if ( (_BYTE)v69 == 14 )
    sub_102B9D0(*(_QWORD *)(a1 + 16), v64);
  sub_278A7A0(a1 + 136, (_BYTE *)a2);
  v72 = *(unsigned int *)(a1 + 656);
  if ( v72 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 660) )
  {
    sub_C8D5F0(a1 + 648, (const void *)(a1 + 664), v72 + 1, 8u, v70, v71);
    v72 = *(unsigned int *)(a1 + 656);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8 * v72) = a2;
  ++*(_DWORD *)(a1 + 656);
  v73 = *(__int64 **)(a1 + 96);
  v74 = *v73;
  v75 = sub_B2BE50(*v73);
  if ( sub_B6EA50(v75)
    || (v99 = sub_B2BE50(v74),
        v100 = sub_B6F970(v99),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v100 + 48LL))(v100)) )
  {
    sub_B174A0((__int64)&v124, (__int64)"gvn", (__int64)"LoadPRE", 7, a2);
    sub_B18290((__int64)&v124, "load eliminated by PRE", 0x16u);
    v118 = (unsigned __int64 *)v120;
    v80 = _mm_loadu_si128(&v127);
    v81 = _mm_loadu_si128(&v129);
    LODWORD(v112) = v125;
    v82 = _mm_loadu_si128(&v130);
    v111 = &unk_49D9D40;
    BYTE4(v112) = BYTE4(v125);
    v114 = v80;
    v113 = v126;
    v116 = v81;
    v115 = v128;
    v119 = 0x400000000LL;
    v117 = v82;
    if ( v132 )
    {
      sub_27900D0((__int64)&v118, (__int64)&v131, v76, v77, v78, v79);
      v124 = (unsigned __int8 *)&unk_49D9D40;
      v97 = v131;
      v121 = v134;
      v122 = v135;
      v123 = v136;
      v111 = &unk_49D9D78;
      v83 = &v131[10 * v132];
      if ( v131 != v83 )
      {
        do
        {
          v83 -= 10;
          v98 = v83[4];
          if ( (unsigned __int64 *)v98 != v83 + 6 )
            j_j___libc_free_0(v98);
          if ( (unsigned __int64 *)*v83 != v83 + 2 )
            j_j___libc_free_0(*v83);
        }
        while ( v97 != v83 );
        v83 = v131;
        if ( v131 == (unsigned __int64 *)v133 )
          goto LABEL_73;
        goto LABEL_72;
      }
    }
    else
    {
      v83 = v131;
      v121 = v134;
      v122 = v135;
      v123 = v136;
      v111 = &unk_49D9D78;
    }
    if ( v83 == (unsigned __int64 *)v133 )
    {
LABEL_73:
      sub_1049740(v73, (__int64)&v111);
      v84 = v118;
      v111 = &unk_49D9D40;
      v85 = &v118[10 * (unsigned int)v119];
      if ( v118 != v85 )
      {
        do
        {
          v85 -= 10;
          v86 = v85[4];
          if ( (unsigned __int64 *)v86 != v85 + 6 )
            j_j___libc_free_0(v86);
          if ( (unsigned __int64 *)*v85 != v85 + 2 )
            j_j___libc_free_0(*v85);
        }
        while ( v84 != v85 );
        v85 = v118;
      }
      if ( v85 != (unsigned __int64 *)v120 )
        _libc_free((unsigned __int64)v85);
      return;
    }
LABEL_72:
    _libc_free((unsigned __int64)v83);
    goto LABEL_73;
  }
}
