// Function: sub_33A4EA0
// Address: 0x33a4ea0
//
void __fastcall sub_33A4EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const __m128i **a5)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rsi
  int v12; // r13d
  char v13; // r15
  int v14; // r9d
  __int64 v15; // rax
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rax
  unsigned __int16 v18; // r13
  __int64 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __m128i *v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned __int16 *v26; // rdx
  int v27; // eax
  __int64 v28; // rdx
  __int16 v29; // ax
  __int64 v30; // rdx
  __int64 (*v31)(); // rax
  const __m128i *v32; // rdi
  __m128i v33; // xmm1
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  const __m128i *v36; // rax
  int v37; // eax
  int v38; // edx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // r15
  __int64 v42; // rax
  int v43; // edx
  __int64 *v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rsi
  unsigned __int16 v47; // r15
  __int64 v48; // r8
  bool v49; // al
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdx
  char v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 (__fastcall *v57)(__int64, __int64, unsigned int); // r13
  __int64 v58; // rax
  _DWORD *v59; // rax
  int v60; // r10d
  int v61; // edx
  unsigned __int16 v62; // ax
  __int32 v63; // edx
  __int64 v64; // rax
  __int64 v65; // r10
  __int64 *v66; // rdi
  __int32 v67; // edx
  __int64 v68; // rax
  _DWORD *v69; // rax
  int v70; // r10d
  int v71; // edx
  unsigned __int16 v72; // ax
  __int64 v73; // rax
  __int32 v74; // edx
  int v75; // esi
  int v76; // eax
  int v77; // r9d
  int v78; // r8d
  int v79; // edx
  int v80; // ecx
  __int32 v81; // edx
  unsigned __int16 v82; // ax
  __int64 v83; // [rsp-10h] [rbp-1D0h]
  __int64 v84; // [rsp-10h] [rbp-1D0h]
  __m128i *v85; // [rsp-8h] [rbp-1C8h]
  __int64 v86; // [rsp+0h] [rbp-1C0h]
  int v87; // [rsp+8h] [rbp-1B8h]
  __int16 v88; // [rsp+12h] [rbp-1AEh]
  __int16 v89; // [rsp+18h] [rbp-1A8h]
  int v90; // [rsp+18h] [rbp-1A8h]
  __int64 v91; // [rsp+18h] [rbp-1A8h]
  __int64 (__fastcall *v92)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-1A8h]
  __int64 v93; // [rsp+20h] [rbp-1A0h]
  __int64 v95; // [rsp+38h] [rbp-188h]
  int v96; // [rsp+38h] [rbp-188h]
  int v97; // [rsp+38h] [rbp-188h]
  __int64 v98; // [rsp+80h] [rbp-140h] BYREF
  __int64 v99; // [rsp+88h] [rbp-138h]
  int v100; // [rsp+9Ch] [rbp-124h] BYREF
  __int64 v101; // [rsp+A0h] [rbp-120h] BYREF
  int v102; // [rsp+A8h] [rbp-118h]
  __m128i v103; // [rsp+B0h] [rbp-110h] BYREF
  __m128i v104; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v105; // [rsp+D0h] [rbp-F0h] BYREF
  unsigned int v106; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v107; // [rsp+E8h] [rbp-D8h]
  __int64 v108; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v109; // [rsp+F8h] [rbp-C8h]
  __int64 v110; // [rsp+100h] [rbp-C0h]
  __int64 v111; // [rsp+108h] [rbp-B8h]
  __int64 v112[4]; // [rsp+110h] [rbp-B0h] BYREF
  __m128i v113; // [rsp+130h] [rbp-90h] BYREF
  __m128i v114; // [rsp+140h] [rbp-80h]
  __m128i v115; // [rsp+150h] [rbp-70h]
  __m128i v116; // [rsp+160h] [rbp-60h]
  __m128i v117; // [rsp+170h] [rbp-50h]
  __m128i v118; // [rsp+180h] [rbp-40h]

  v98 = a3;
  v7 = *(_QWORD *)a1;
  v8 = *(_DWORD *)(a1 + 848);
  v99 = a4;
  v101 = 0;
  v102 = v8;
  if ( v7 )
  {
    if ( &v101 != (__int64 *)(v7 + 48) )
    {
      v9 = *(_QWORD *)(v7 + 48);
      v101 = v9;
      if ( v9 )
        sub_B96E90((__int64)&v101, v9, 1);
    }
  }
  v93 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v95 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  LOWORD(v10) = sub_B5A5E0(a2);
  v11 = a2;
  v12 = v10;
  v13 = BYTE1(v10);
  sub_B91FC0(v112, a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 || (v11 = 29, !sub_B91C10(a2, 29)) || (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
  {
    v14 = 0;
    if ( v13 )
      goto LABEL_9;
LABEL_27:
    v47 = v98;
    v48 = *(_QWORD *)(a1 + 864);
    if ( (_WORD)v98 )
    {
      if ( (unsigned __int16)(v98 - 17) <= 0xD3u )
      {
        v52 = 0;
        v47 = word_4456580[(unsigned __int16)v98 - 1];
        goto LABEL_30;
      }
    }
    else
    {
      v86 = *(_QWORD *)(a1 + 864);
      v87 = v14;
      v49 = sub_30070B0((__int64)&v98);
      v14 = v87;
      v48 = v86;
      if ( v49 )
      {
        v82 = sub_3009970((__int64)&v98, v11, v50, v51, v86);
        v14 = v87;
        v48 = v86;
        v47 = v82;
        goto LABEL_30;
      }
    }
    v52 = v99;
LABEL_30:
    v90 = v14;
    v53 = sub_33CC4A0(v48, v47, v52);
    v14 = v90;
    LOBYTE(v12) = v53;
    goto LABEL_9;
  }
  v11 = 4;
  v14 = sub_B91C10(a2, 4);
  if ( !v13 )
    goto LABEL_27;
LABEL_9:
  v15 = *(_QWORD *)(v95 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
  {
    v15 = **(_QWORD **)(v15 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
      v15 = **(_QWORD **)(v15 + 16);
  }
  v16 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  v114.m128i_i32[0] = *(_DWORD *)(v15 + 8) >> 8;
  v114.m128i_i8[4] = 0;
  v113 = 0u;
  v17 = sub_2E7BD70(v16, 1u, -1, v12, (int)v112, v14, 0, v114.m128i_i64[0], 1u, 0, 0);
  v18 = v98;
  v103.m128i_i64[0] = 0;
  v103.m128i_i32[2] = 0;
  v19 = v17;
  v104.m128i_i64[0] = 0;
  v104.m128i_i32[2] = 0;
  v105.m128i_i64[0] = 0;
  v105.m128i_i32[2] = 0;
  if ( (_WORD)v98 )
  {
    if ( (unsigned __int16)(v98 - 17) > 0xD3u )
    {
LABEL_14:
      v20 = v99;
      goto LABEL_15;
    }
    v20 = 0;
    v18 = word_4456580[(unsigned __int16)v98 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v98) )
      goto LABEL_14;
    v18 = sub_3009970((__int64)&v98, 1, v54, v55, v56);
  }
LABEL_15:
  v113.m128i_i16[0] = v18;
  v113.m128i_i64[1] = v20;
  if ( v18 )
  {
    if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
      BUG();
    v21 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
  }
  else
  {
    v21 = sub_3007260((__int64)&v113);
    v110 = v21;
    v111 = v22;
  }
  v23 = &v103;
  if ( !(unsigned __int8)sub_339D300(
                           v95,
                           (__int64)&v103,
                           (__int64)&v104,
                           &v100,
                           (__int64)&v105,
                           a1,
                           *(_QWORD *)(a2 + 40),
                           (unsigned __int64)(v21 + 7) >> 3) )
  {
    v91 = *(_QWORD *)(a1 + 864);
    v57 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v93 + 32LL);
    v58 = sub_2E79000(*(__int64 **)(v91 + 40));
    if ( v57 == sub_2D42F30 )
    {
      v59 = sub_AE2980(v58, 0);
      v60 = v91;
      v61 = v59[1];
      v62 = 2;
      if ( v61 != 1 )
      {
        v62 = 3;
        if ( v61 != 2 )
        {
          v62 = 4;
          if ( v61 != 4 )
          {
            v62 = 5;
            if ( v61 != 8 )
            {
              v62 = 6;
              if ( v61 != 16 )
              {
                v62 = 7;
                if ( v61 != 32 )
                {
                  v62 = 8;
                  if ( v61 != 64 )
                    v62 = 9 * (v61 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v62 = v57(v93, v58, 0);
      v60 = v91;
    }
    v103.m128i_i64[0] = sub_3400BD0(v60, 0, (unsigned int)&v101, v62, 0, 0, 0);
    v103.m128i_i32[2] = v63;
    v64 = sub_338B750(a1, v95);
    v100 = 0;
    v65 = *(_QWORD *)(a1 + 864);
    v104.m128i_i64[0] = v64;
    v66 = *(__int64 **)(v65 + 40);
    v97 = v65;
    v104.m128i_i32[2] = v67;
    v92 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v93 + 32LL);
    v68 = sub_2E79000(v66);
    if ( v92 == sub_2D42F30 )
    {
      v69 = sub_AE2980(v68, 0);
      v70 = v97;
      v71 = v69[1];
      v72 = 2;
      if ( v71 != 1 )
      {
        v72 = 3;
        if ( v71 != 2 )
        {
          v72 = 4;
          if ( v71 != 4 )
          {
            v72 = 5;
            if ( v71 != 8 )
            {
              v72 = 6;
              if ( v71 != 16 )
              {
                v72 = 7;
                if ( v71 != 32 )
                {
                  v72 = 8;
                  if ( v71 != 64 )
                    v72 = 9 * (v71 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v72 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64 (__fastcall *)(__int64, __int64, unsigned int), __int64, __m128i *))v92)(
              v93,
              v68,
              0,
              v92,
              v83,
              v85);
      v70 = v97;
    }
    v73 = sub_3400BD0(v70, 1, (unsigned int)&v101, v72, 0, 1, 0);
    v24 = v84;
    v23 = v85;
    v105.m128i_i64[0] = v73;
    v105.m128i_i32[2] = v74;
  }
  v26 = (unsigned __int16 *)(*(_QWORD *)(v104.m128i_i64[0] + 48) + 16LL * v104.m128i_u32[2]);
  v27 = *v26;
  v28 = *((_QWORD *)v26 + 1);
  LOWORD(v106) = v27;
  v107 = v28;
  if ( (_WORD)v27 )
  {
    v29 = word_4456580[v27 - 1];
    v30 = 0;
  }
  else
  {
    v29 = sub_3009970((__int64)&v106, (__int64)v23, v28, v24, v25);
  }
  LOWORD(v108) = v29;
  v109 = v30;
  v31 = *(__int64 (**)())(*(_QWORD *)v93 + 696LL);
  if ( v31 != sub_2FE3240
    && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64 *))v31)(v93, v106, v107, &v108) )
  {
    if ( (_WORD)v106 )
    {
      v75 = word_4456340[(unsigned __int16)v106 - 1];
      if ( (unsigned __int16)(v106 - 176) > 0x34u )
        LOWORD(v76) = sub_2D43050(v108, v75);
      else
        LOWORD(v76) = sub_2D43AD0(v108, v75);
      v78 = 0;
    }
    else
    {
      v76 = sub_3009490((unsigned __int16 *)&v106, v108, v109);
      v88 = HIWORD(v76);
      v78 = v79;
    }
    HIWORD(v80) = v88;
    LOWORD(v80) = v76;
    v104.m128i_i64[0] = sub_33FAF80(*(_QWORD *)(a1 + 864), 213, (unsigned int)&v101, v80, v78, v77);
    v104.m128i_i32[2] = v81;
  }
  v32 = *(const __m128i **)(a1 + 864);
  v33 = _mm_load_si128(&v103);
  v34 = _mm_load_si128(&v104);
  v35 = _mm_load_si128(&v105);
  v89 = v100;
  v36 = *a5;
  v113 = _mm_loadu_si128(v32 + 24);
  v114 = v33;
  v115 = v34;
  v116 = v35;
  v117 = _mm_loadu_si128(v36 + 1);
  v118 = _mm_loadu_si128(v36 + 2);
  v37 = sub_33E5110(v32, (unsigned int)v98, v99, 1, 0);
  v41 = sub_33E79D0((_DWORD)v32, v37, v38, v98, v99, (unsigned int)&v101, (__int64)&v113, 6, v19, v89);
  v42 = *(unsigned int *)(a1 + 136);
  v96 = v43;
  if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
  {
    sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v42 + 1, 0x10u, v39, v40);
    v42 = *(unsigned int *)(a1 + 136);
  }
  v44 = (__int64 *)(*(_QWORD *)(a1 + 128) + 16 * v42);
  *v44 = v41;
  v44[1] = 1;
  ++*(_DWORD *)(a1 + 136);
  v113.m128i_i64[0] = a2;
  v45 = sub_337DC20(a1 + 8, v113.m128i_i64);
  *v45 = v41;
  v46 = v101;
  *((_DWORD *)v45 + 2) = v96;
  if ( v46 )
    sub_B91220((__int64)&v101, v46);
}
