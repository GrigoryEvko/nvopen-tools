// Function: sub_33A5840
// Address: 0x33a5840
//
void __fastcall sub_33A5840(__int64 a1, __int64 a2, const __m128i **a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rax
  __int16 v10; // dx
  __int64 v11; // rax
  int v12; // eax
  int v13; // r13d
  __int64 v14; // rax
  __int32 v15; // eax
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rax
  unsigned __int16 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __m128i *v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned __int16 *v25; // rdx
  int v26; // eax
  __int64 v27; // rdx
  __int16 v28; // ax
  __int64 v29; // rdx
  __int64 (*v30)(); // rax
  __int64 v31; // r15
  __int64 v32; // rax
  __m128i v33; // xmm1
  __m128i v34; // xmm2
  __int64 v35; // rdx
  __int64 v36; // rdi
  const __m128i *v37; // rax
  __m128i v38; // xmm3
  __m128i v39; // xmm0
  __int64 v40; // rcx
  int v41; // eax
  int v42; // edx
  __int64 v43; // rax
  int v44; // edx
  __int64 v45; // r8
  __int64 v46; // r15
  _QWORD *v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  unsigned __int16 v52; // dx
  __int64 v53; // r8
  bool v54; // al
  __int64 v55; // rcx
  __int64 v56; // r9
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
  __int64 v83; // rdx
  __int64 v84; // [rsp-10h] [rbp-1E0h]
  __int64 v85; // [rsp-10h] [rbp-1E0h]
  __m128i *v86; // [rsp-8h] [rbp-1D8h]
  unsigned int v87; // [rsp+Ch] [rbp-1C4h]
  __int16 v88; // [rsp+12h] [rbp-1BEh]
  __int16 v89; // [rsp+18h] [rbp-1B8h]
  __int64 v90; // [rsp+18h] [rbp-1B8h]
  __int64 v91; // [rsp+18h] [rbp-1B8h]
  __int64 (__fastcall *v92)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-1B8h]
  char v93; // [rsp+20h] [rbp-1B0h]
  __int64 v94; // [rsp+20h] [rbp-1B0h]
  __int64 v96; // [rsp+28h] [rbp-1A8h]
  __int64 v97; // [rsp+30h] [rbp-1A0h]
  int v98; // [rsp+30h] [rbp-1A0h]
  int v99; // [rsp+30h] [rbp-1A0h]
  int v100; // [rsp+8Ch] [rbp-144h] BYREF
  __int64 v101; // [rsp+90h] [rbp-140h] BYREF
  int v102; // [rsp+98h] [rbp-138h]
  int v103; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v104; // [rsp+A8h] [rbp-128h]
  __m128i v105; // [rsp+B0h] [rbp-120h] BYREF
  __m128i v106; // [rsp+C0h] [rbp-110h] BYREF
  __m128i v107; // [rsp+D0h] [rbp-100h] BYREF
  unsigned int v108; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v109; // [rsp+E8h] [rbp-E8h]
  __int64 v110; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v111; // [rsp+F8h] [rbp-D8h]
  __int64 v112; // [rsp+100h] [rbp-D0h]
  __int64 v113; // [rsp+108h] [rbp-C8h]
  __int64 v114[4]; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v115; // [rsp+130h] [rbp-A0h] BYREF
  __int64 v116; // [rsp+138h] [rbp-98h]
  __m128i v117; // [rsp+140h] [rbp-90h]
  __m128i v118; // [rsp+150h] [rbp-80h]
  __m128i v119; // [rsp+160h] [rbp-70h]
  __m128i v120; // [rsp+170h] [rbp-60h]
  __m128i v121; // [rsp+180h] [rbp-50h]
  __m128i v122; // [rsp+190h] [rbp-40h]

  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 848);
  v101 = 0;
  v102 = v6;
  if ( v5 )
  {
    if ( &v101 != (__int64 *)(v5 + 48) )
    {
      v7 = *(_QWORD *)(v5 + 48);
      v101 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v101, v7, 1);
    }
  }
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v97 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v9 = *(_QWORD *)((*a3)->m128i_i64[0] + 48) + 16LL * (*a3)->m128i_u32[2];
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOWORD(v103) = v10;
  v104 = v11;
  LOWORD(v12) = sub_B5A5E0(a2);
  v13 = v12;
  v93 = BYTE1(v12);
  sub_B91FC0(v114, a2);
  if ( !v93 )
  {
    v52 = v103;
    v53 = *(_QWORD *)(a1 + 864);
    if ( (_WORD)v103 )
    {
      if ( (unsigned __int16)(v103 - 17) <= 0xD3u )
      {
        v56 = 0;
        v52 = word_4456580[(unsigned __int16)v103 - 1];
        goto LABEL_30;
      }
    }
    else
    {
      v87 = (unsigned __int16)v103;
      v90 = *(_QWORD *)(a1 + 864);
      v54 = sub_30070B0((__int64)&v103);
      v53 = v90;
      v52 = v87;
      if ( v54 )
      {
        v82 = sub_3009970((__int64)&v103, a2, v87, v55, v90);
        v53 = v90;
        v56 = v83;
        v52 = v82;
        goto LABEL_30;
      }
    }
    v56 = v104;
LABEL_30:
    LOBYTE(v13) = sub_33CC4A0(v53, v52, v56);
  }
  v14 = *(_QWORD *)(v97 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
  {
    v14 = **(_QWORD **)(v14 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
  }
  v15 = *(_DWORD *)(v14 + 8) >> 8;
  v16 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  v116 = 0;
  v117.m128i_i32[0] = v15;
  v117.m128i_i8[4] = 0;
  v115 = 0;
  v17 = sub_2E7BD70(v16, 2u, -1, v13, (int)v114, 0, 0, v117.m128i_i64[0], 1u, 0, 0);
  v18 = v103;
  v105.m128i_i64[0] = 0;
  v94 = v17;
  v105.m128i_i32[2] = 0;
  v106.m128i_i64[0] = 0;
  v106.m128i_i32[2] = 0;
  v107.m128i_i64[0] = 0;
  v107.m128i_i32[2] = 0;
  if ( (_WORD)v103 )
  {
    if ( (unsigned __int16)(v103 - 17) > 0xD3u )
    {
LABEL_11:
      v19 = v104;
      goto LABEL_12;
    }
    v19 = 0;
    v18 = word_4456580[(unsigned __int16)v103 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v103) )
      goto LABEL_11;
    v18 = sub_3009970((__int64)&v103, 2, v49, v50, v51);
  }
LABEL_12:
  LOWORD(v115) = v18;
  v116 = v19;
  if ( v18 )
  {
    if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
      BUG();
    v20 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
  }
  else
  {
    v20 = sub_3007260((__int64)&v115);
    v112 = v20;
    v113 = v21;
  }
  v22 = &v105;
  if ( !(unsigned __int8)sub_339D300(
                           v97,
                           (__int64)&v105,
                           (__int64)&v106,
                           &v100,
                           (__int64)&v107,
                           a1,
                           *(_QWORD *)(a2 + 40),
                           (unsigned __int64)(v20 + 7) >> 3) )
  {
    v57 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v8 + 32LL);
    v91 = *(_QWORD *)(a1 + 864);
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
      v62 = v57(v8, v58, 0);
      v60 = v91;
    }
    v105.m128i_i64[0] = sub_3400BD0(v60, 0, (unsigned int)&v101, v62, 0, 0, 0);
    v105.m128i_i32[2] = v63;
    v64 = sub_338B750(a1, v97);
    v65 = *(_QWORD *)(a1 + 864);
    v100 = 0;
    v66 = *(__int64 **)(v65 + 40);
    v106.m128i_i64[0] = v64;
    v99 = v65;
    v106.m128i_i32[2] = v67;
    v92 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v8 + 32LL);
    v68 = sub_2E79000(v66);
    if ( v92 == sub_2D42F30 )
    {
      v69 = sub_AE2980(v68, 0);
      v70 = v99;
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
              v8,
              v68,
              0,
              v92,
              v84,
              v86);
      v70 = v99;
    }
    v73 = sub_3400BD0(v70, 1, (unsigned int)&v101, v72, 0, 1, 0);
    v23 = v85;
    v22 = v86;
    v107.m128i_i64[0] = v73;
    v107.m128i_i32[2] = v74;
  }
  v25 = (unsigned __int16 *)(*(_QWORD *)(v106.m128i_i64[0] + 48) + 16LL * v106.m128i_u32[2]);
  v26 = *v25;
  v27 = *((_QWORD *)v25 + 1);
  LOWORD(v108) = v26;
  v109 = v27;
  if ( (_WORD)v26 )
  {
    v28 = word_4456580[v26 - 1];
    v29 = 0;
  }
  else
  {
    v28 = sub_3009970((__int64)&v108, (__int64)v22, v27, v23, v24);
  }
  LOWORD(v110) = v28;
  v111 = v29;
  v30 = *(__int64 (**)())(*(_QWORD *)v8 + 696LL);
  if ( v30 != sub_2FE3240
    && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64 *))v30)(v8, v108, v109, &v110) )
  {
    if ( (_WORD)v108 )
    {
      v75 = word_4456340[(unsigned __int16)v108 - 1];
      if ( (unsigned __int16)(v108 - 176) > 0x34u )
        LOWORD(v76) = sub_2D43050(v110, v75);
      else
        LOWORD(v76) = sub_2D43AD0(v110, v75);
      v78 = 0;
    }
    else
    {
      v76 = sub_3009490((unsigned __int16 *)&v108, v110, v111);
      v88 = HIWORD(v76);
      v78 = v79;
    }
    HIWORD(v80) = v88;
    LOWORD(v80) = v76;
    v106.m128i_i64[0] = sub_33FAF80(*(_QWORD *)(a1 + 864), 213, (unsigned int)&v101, v80, v78, v77);
    v106.m128i_i32[2] = v81;
  }
  v31 = *(_QWORD *)(a1 + 864);
  v89 = v100;
  v32 = sub_33738A0(a1);
  v33 = _mm_loadu_si128(&v105);
  v34 = _mm_loadu_si128(&v106);
  v115 = v32;
  v116 = v35;
  v36 = *(_QWORD *)(a1 + 864);
  v37 = *a3;
  v38 = _mm_loadu_si128(&v107);
  v39 = _mm_loadu_si128(*a3);
  v118 = v33;
  v119 = v34;
  v117 = v39;
  v120 = v38;
  v121 = _mm_loadu_si128(v37 + 2);
  v122 = _mm_loadu_si128(v37 + 3);
  v41 = sub_33ED250(v36, 1, 0, v40);
  v43 = sub_33E6FD0(v31, v41, v42, v103, v104, (unsigned int)&v101, (__int64)&v115, 7, v94, v89);
  v45 = *(_QWORD *)(a1 + 864);
  v98 = v44;
  v46 = v43;
  if ( v43 )
  {
    v96 = *(_QWORD *)(a1 + 864);
    nullsub_1875(v43, v45, 0);
    *(_QWORD *)(v96 + 384) = v46;
    *(_DWORD *)(v96 + 392) = v98;
    sub_33E2B60(v96, 0);
  }
  else
  {
    *(_QWORD *)(v45 + 384) = 0;
    *(_DWORD *)(v45 + 392) = v44;
  }
  v115 = a2;
  v47 = sub_337DC20(a1 + 8, &v115);
  *v47 = v46;
  v48 = v101;
  *((_DWORD *)v47 + 2) = v98;
  if ( v48 )
    sub_B91220((__int64)&v101, v48);
}
