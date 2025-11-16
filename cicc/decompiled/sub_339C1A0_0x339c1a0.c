// Function: sub_339C1A0
// Address: 0x339c1a0
//
void __fastcall sub_339C1A0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  int v6; // eax
  int v7; // ecx
  unsigned int v8; // eax
  __int64 v9; // rdx
  int v10; // edi
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 *v16; // rdi
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r13
  unsigned __int64 v21; // r12
  unsigned __int8 v22; // dl
  unsigned __int8 v23; // al
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 (__fastcall *v28)(__int64, __int64, unsigned int); // rax
  __int64 v29; // rsi
  _DWORD *v30; // rax
  __int16 v31; // dx
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 v36; // r11
  unsigned __int16 v37; // r15
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int128 v41; // rax
  int v42; // r9d
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r13
  unsigned int v47; // edx
  __int64 v48; // r12
  _QWORD *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rdx
  int v56; // eax
  int v57; // edx
  int v58; // r9d
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r13
  __int64 v62; // r12
  _QWORD *v63; // rax
  __int64 v64; // r13
  __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // rdx
  __int128 v68; // rax
  int v69; // r9d
  unsigned int v70; // edx
  __int64 (*v71)(); // rax
  unsigned __int8 v72; // cl
  __int128 v73; // rax
  unsigned __int16 *v74; // rsi
  __int64 v75; // r8
  int v76; // ecx
  __int64 v77; // rax
  unsigned int v78; // edx
  unsigned int v79; // r12d
  __int64 v80; // r13
  __int128 v81; // rax
  unsigned __int16 *v82; // rsi
  int v83; // ecx
  __int64 v84; // r8
  int v85; // r9d
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rdx
  bool v91; // al
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  unsigned __int16 v95; // ax
  __int64 v96; // rdx
  __int128 v97; // [rsp-20h] [rbp-120h]
  unsigned __int8 v98; // [rsp+7h] [rbp-F9h]
  __int64 v99; // [rsp+8h] [rbp-F8h]
  __int64 v100; // [rsp+8h] [rbp-F8h]
  char v101; // [rsp+8h] [rbp-F8h]
  __int64 v102; // [rsp+8h] [rbp-F8h]
  __int64 v103; // [rsp+8h] [rbp-F8h]
  char v104; // [rsp+10h] [rbp-F0h]
  int v105; // [rsp+10h] [rbp-F0h]
  __int64 v106; // [rsp+10h] [rbp-F0h]
  char v107; // [rsp+18h] [rbp-E8h]
  unsigned int v108; // [rsp+18h] [rbp-E8h]
  __int64 v109; // [rsp+18h] [rbp-E8h]
  __int64 v110; // [rsp+18h] [rbp-E8h]
  __m128i v111; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v112; // [rsp+30h] [rbp-D0h]
  __int64 v113; // [rsp+38h] [rbp-C8h]
  __int64 v114; // [rsp+40h] [rbp-C0h]
  __int64 v115; // [rsp+48h] [rbp-B8h]
  __int64 v116; // [rsp+50h] [rbp-B0h]
  __int64 v117; // [rsp+58h] [rbp-A8h]
  __int64 v118; // [rsp+68h] [rbp-98h] BYREF
  __int64 v119; // [rsp+70h] [rbp-90h] BYREF
  int v120; // [rsp+78h] [rbp-88h]
  unsigned int v121; // [rsp+80h] [rbp-80h] BYREF
  __int64 v122; // [rsp+88h] [rbp-78h]
  __int64 v123; // [rsp+90h] [rbp-70h]
  __int64 v124; // [rsp+98h] [rbp-68h]
  unsigned __int64 v125; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v126; // [rsp+A8h] [rbp-58h]
  __m128i v127; // [rsp+B0h] [rbp-50h]
  __int64 v128; // [rsp+C0h] [rbp-40h]
  __int64 v129; // [rsp+C8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 960);
  v5 = *(_QWORD *)(v4 + 256);
  v6 = *(_DWORD *)(v4 + 272);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = *(_QWORD *)(v5 + 16LL * v8);
    if ( a2 == v9 )
      return;
    v10 = 1;
    while ( v9 != -4096 )
    {
      v8 = v7 & (v10 + v8);
      v9 = *(_QWORD *)(v5 + 16LL * v8);
      if ( a2 == v9 )
        return;
      ++v10;
    }
  }
  v11 = *(_DWORD *)(a1 + 848);
  v12 = *(_QWORD *)a1;
  v119 = 0;
  v120 = v11;
  if ( v12 )
  {
    if ( &v119 != (__int64 *)(v12 + 48) )
    {
      v13 = *(_QWORD *)(v12 + 48);
      v119 = v13;
      if ( v13 )
        sub_B96E90((__int64)&v119, v13, 1);
    }
  }
  v14 = *(_QWORD *)(a1 + 864);
  v15 = *(_QWORD *)(v14 + 16);
  v16 = *(__int64 **)(v14 + 40);
  v111.m128i_i64[0] = *(_QWORD *)(a2 + 72);
  v99 = v15;
  v17 = sub_2E79000(v16);
  v107 = sub_AE5020(v17, v111.m128i_i64[0]);
  v18 = sub_9208B0(v17, v111.m128i_i64[0]);
  v126 = v19;
  v125 = v18;
  v104 = v19;
  v20 = (((unsigned __int64)(v18 + 7) >> 3) + (1LL << v107) - 1) >> v107 << v107;
  _BitScanReverse64(&v21, 1LL << *(_WORD *)(a2 + 2));
  v22 = sub_AE5260(v17, v111.m128i_i64[0]);
  v23 = 63 - (v21 ^ 0x3F);
  if ( v22 > v23 )
    v23 = v22;
  v98 = v23;
  v24 = sub_338B750(a1, *(_QWORD *)(a2 - 32));
  v111.m128i_i64[1] = v25;
  v26 = v24;
  v108 = v25;
  v27 = *(_QWORD *)(a2 + 8);
  v111.m128i_i64[0] = v24;
  v28 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v99 + 32LL);
  v29 = *(_DWORD *)(v27 + 8) >> 8;
  if ( v28 == sub_2D42F30 )
  {
    v30 = sub_AE2980(v17, v29);
    v31 = 2;
    v32 = v30[1];
    if ( v32 != 1 )
    {
      v31 = 3;
      if ( v32 != 2 )
      {
        v31 = 4;
        if ( v32 != 4 )
        {
          v31 = 5;
          if ( v32 != 8 )
          {
            v31 = 6;
            if ( v32 != 16 )
            {
              v31 = 7;
              if ( v32 != 32 )
              {
                v31 = 8;
                if ( v32 != 64 )
                  v31 = 9 * (v32 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v96 = (unsigned int)v29;
    v29 = v17;
    v31 = v28(v99, v17, v96);
  }
  LOWORD(v121) = v31;
  v122 = 0;
  v33 = *(_QWORD *)(v26 + 48) + 16LL * v108;
  if ( v31 != *(_WORD *)v33 || *(_QWORD *)(v33 + 8) && !v31 )
  {
    v29 = v111.m128i_i64[0];
    v34 = sub_33FB310(*(_QWORD *)(a1 + 864), v111.m128i_i64[0], v111.m128i_i64[1], &v119, v121, v122);
    v108 = v35;
    v26 = v34;
  }
  v36 = *(_QWORD *)(a1 + 864);
  if ( !v104 )
  {
    v65 = sub_3400BD0(v36, v20, (unsigned int)&v119, 8, 0, 0, 0);
    v66 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v68 = sub_33FB310(v66, v65, v67, &v119, v121, v122);
    v111.m128i_i64[0] = v26;
    v111.m128i_i64[1] = v108 | v111.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v46 = sub_3406EB0(v66, 58, (unsigned int)&v119, v121, v122, v69, *(_OWORD *)&v111, v68);
    v48 = v70;
    goto LABEL_34;
  }
  v37 = v121;
  if ( (_WORD)v121 )
  {
    if ( (unsigned __int16)(v121 - 17) <= 0xD3u )
    {
      v37 = word_4456580[(unsigned __int16)v121 - 1];
      v38 = 0;
      goto LABEL_27;
    }
  }
  else
  {
    v102 = *(_QWORD *)(a1 + 864);
    v91 = sub_30070B0((__int64)&v121);
    v36 = v102;
    if ( v91 )
    {
      v95 = sub_3009970((__int64)&v121, v29, v92, v93, v94);
      v36 = v102;
      v37 = v95;
      goto LABEL_27;
    }
  }
  v38 = v122;
LABEL_27:
  LOWORD(v125) = v37;
  v126 = v38;
  if ( v37 )
  {
    if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
      BUG();
    v39 = *(_QWORD *)&byte_444C4A0[16 * v37 - 16];
  }
  else
  {
    v100 = v36;
    v39 = sub_3007260((__int64)&v125);
    v36 = v100;
    v123 = v39;
    v124 = v40;
  }
  LODWORD(v126) = v39;
  if ( (unsigned int)v39 > 0x40 )
  {
    v103 = v36;
    sub_C43690((__int64)&v125, v20, 0);
    v36 = v103;
  }
  else
  {
    v125 = v20;
  }
  v105 = v36;
  *(_QWORD *)&v41 = sub_3401900(v36, &v119, v121, v122, &v125, 1);
  v111.m128i_i64[0] = v26;
  v111.m128i_i64[1] = v108 | v111.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v46 = sub_3406EB0(v105, 58, (unsigned int)&v119, v121, v122, v42, *(_OWORD *)&v111, v41);
  v48 = v47;
  if ( (unsigned int)v126 > 0x40 && v125 )
    j_j___libc_free_0_0(v125);
LABEL_34:
  v49 = *(_QWORD **)(a1 + 864);
  if ( (unsigned int)(*(_DWORD *)(*v49 + 544LL) - 42) > 1 )
  {
    v71 = *(__int64 (**)())(**(_QWORD **)(v49[5] + 16LL) + 136LL);
    if ( v71 == sub_2DD19D0 )
      BUG();
    v101 = 1;
    v72 = *(_BYTE *)(v71() + 12);
    if ( v72 >= v98 )
    {
      v98 = 0;
      v101 = 0;
    }
    v106 = 1LL << v72;
    v109 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v73 = sub_3400BD0(v109, (unsigned int)(1LL << v72) - 1, (unsigned int)&v119, v121, v122, 0, 0);
    v74 = (unsigned __int16 *)(*(_QWORD *)(v46 + 48) + 16 * v48);
    v111.m128i_i64[0] = v46;
    v75 = *((_QWORD *)v74 + 1);
    v76 = *v74;
    v111.m128i_i64[1] = v48 | v111.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v77 = sub_3405C90(v109, 56, (unsigned int)&v119, v76, v75, 1, __PAIR128__(v111.m128i_u64[1], v46), v73);
    v79 = v78;
    v80 = v77;
    v110 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v81 = sub_3401400(v110, -(int)v106, (unsigned int)&v119, v121, v122, 0, 0);
    v82 = (unsigned __int16 *)(*(_QWORD *)(v80 + 48) + 16LL * v79);
    v111.m128i_i64[0] = v80;
    v83 = *v82;
    v84 = *((_QWORD *)v82 + 1);
    v111.m128i_i64[1] = v79 | v111.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v46 = sub_3406EB0(v110, 186, (unsigned int)&v119, v83, v84, v85, __PAIR128__(v111.m128i_u64[1], v80), v81);
    LODWORD(v110) = v86;
    v48 = (unsigned int)v86;
    v111.m128i_i64[0] = v46;
    v125 = sub_33738B0(a1, 186, v86, v87, v88, v89);
    v126 = v90;
    v51 = *(_QWORD *)(a1 + 864);
    v111.m128i_i64[1] = (unsigned int)v110 | v111.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v127 = _mm_load_si128(&v111);
    if ( !v101 )
    {
      LODWORD(v53) = 0;
      goto LABEL_37;
    }
  }
  else
  {
    v50 = sub_33738B0(a1, 58, (__int64)v49, v43, v44, v45);
    v111.m128i_i64[0] = v46;
    v51 = *(_QWORD *)(a1 + 864);
    v125 = v50;
    v126 = v52;
    v111.m128i_i64[1] = v48 | v111.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v127 = _mm_load_si128(&v111);
  }
  v53 = 1LL << v98;
LABEL_37:
  v128 = sub_3400BD0(v51, v53, (unsigned int)&v119, v121, v122, 0, 0);
  v54 = *(_QWORD *)(a1 + 864);
  v129 = v55;
  v56 = sub_33E5110(
          v54,
          *(unsigned __int16 *)(*(_QWORD *)(v46 + 48) + 16 * v48),
          *(_QWORD *)(*(_QWORD *)(v46 + 48) + 16 * v48 + 8),
          1,
          0);
  *((_QWORD *)&v97 + 1) = 3;
  *(_QWORD *)&v97 = &v125;
  v59 = sub_3411630(*(_QWORD *)(a1 + 864), 300, (unsigned int)&v119, v56, v57, v58, v97);
  v61 = v60;
  v62 = v59;
  v118 = a2;
  v63 = sub_337DC20(a1 + 8, &v118);
  v117 = v61;
  v116 = v62;
  *v63 = v62;
  *((_DWORD *)v63 + 2) = v117;
  v64 = *(_QWORD *)(a1 + 864);
  if ( v62 )
  {
    nullsub_1875(v62, v64, 0);
    v115 = 1;
    v114 = v62;
    *(_QWORD *)(v64 + 384) = v62;
    *(_DWORD *)(v64 + 392) = v115;
    sub_33E2B60(v64, 0);
  }
  else
  {
    v113 = 1;
    v112 = 0;
    *(_QWORD *)(v64 + 384) = 0;
    *(_DWORD *)(v64 + 392) = v113;
  }
  if ( v119 )
    sub_B91220((__int64)&v119, v119);
}
