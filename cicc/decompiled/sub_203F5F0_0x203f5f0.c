// Function: sub_203F5F0
// Address: 0x203f5f0
//
void __fastcall sub_203F5F0(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, __m128i a6)
{
  const __m128i *v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // r13
  int v11; // r15d
  __int64 v12; // rax
  __int16 v13; // si
  __m128i v14; // xmm1
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int v26; // eax
  const void **v27; // rdx
  unsigned int v28; // eax
  __int64 *v29; // r12
  __int64 v30; // rax
  unsigned int v31; // edx
  unsigned __int8 v32; // al
  __int128 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r9
  __int64 v39; // r8
  __int64 v40; // rdx
  __int64 *v41; // rdx
  unsigned int v42; // r13d
  int v43; // eax
  unsigned int v44; // r15d
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r9
  __int64 v49; // r8
  __int64 v50; // rdx
  __int64 *v51; // rdx
  __int64 *v52; // r12
  __int64 v53; // rax
  unsigned int v54; // edi
  __int64 v55; // r10
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 *v58; // r12
  unsigned int v59; // edx
  __int64 v60; // rax
  unsigned int v61; // edx
  char v62; // al
  __int128 v63; // rax
  __int64 *v64; // r8
  __int64 v65; // rdx
  __int64 v66; // r9
  __int64 v67; // rcx
  _QWORD *v68; // rdi
  unsigned __int64 v69; // rsi
  __int64 v70; // r12
  char v71; // r10
  __int64 v72; // rdx
  bool v73; // zf
  __int128 v74; // [rsp-20h] [rbp-1A0h]
  __int64 v75; // [rsp+0h] [rbp-180h]
  __int64 v76; // [rsp+8h] [rbp-178h]
  int v77; // [rsp+10h] [rbp-170h]
  __int16 v78; // [rsp+24h] [rbp-15Ch]
  __int64 v79; // [rsp+28h] [rbp-158h]
  unsigned int v80; // [rsp+30h] [rbp-150h]
  unsigned int v81; // [rsp+34h] [rbp-14Ch]
  __int64 v83; // [rsp+40h] [rbp-140h]
  __int64 v84; // [rsp+48h] [rbp-138h]
  __int64 v85; // [rsp+50h] [rbp-130h]
  unsigned __int64 v86; // [rsp+58h] [rbp-128h]
  unsigned int v87; // [rsp+60h] [rbp-120h]
  __int64 v88; // [rsp+60h] [rbp-120h]
  unsigned int v89; // [rsp+68h] [rbp-118h]
  __int64 (__fastcall *v90)(__int64, __int64); // [rsp+68h] [rbp-118h]
  unsigned int v91; // [rsp+70h] [rbp-110h]
  unsigned __int32 v92; // [rsp+78h] [rbp-108h]
  __int64 v93; // [rsp+78h] [rbp-108h]
  __int64 v94; // [rsp+80h] [rbp-100h]
  unsigned int v95; // [rsp+80h] [rbp-100h]
  __int64 v96; // [rsp+88h] [rbp-F8h]
  int v97; // [rsp+90h] [rbp-F0h]
  __int64 (__fastcall *v98)(__int64, __int64); // [rsp+A0h] [rbp-E0h]
  unsigned __int64 v100; // [rsp+B8h] [rbp-C8h]
  __int16 v101; // [rsp+C0h] [rbp-C0h]
  __int64 *v102; // [rsp+C0h] [rbp-C0h]
  __int64 v103; // [rsp+C0h] [rbp-C0h]
  __int64 v104; // [rsp+C0h] [rbp-C0h]
  __int64 v105; // [rsp+C8h] [rbp-B8h]
  __int64 v106; // [rsp+C8h] [rbp-B8h]
  __int64 v107; // [rsp+C8h] [rbp-B8h]
  __int64 v108; // [rsp+D0h] [rbp-B0h] BYREF
  int v109; // [rsp+D8h] [rbp-A8h]
  char v110[8]; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v111; // [rsp+E8h] [rbp-98h]
  char v112[8]; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v113; // [rsp+F8h] [rbp-88h]
  unsigned int v114; // [rsp+100h] [rbp-80h] BYREF
  const void **v115; // [rsp+108h] [rbp-78h]
  __m128i v116; // [rsp+110h] [rbp-70h] BYREF
  __int64 v117; // [rsp+120h] [rbp-60h]
  __int128 v118; // [rsp+130h] [rbp-50h]
  __int64 v119; // [rsp+140h] [rbp-40h]

  v8 = *(const __m128i **)(a3 + 32);
  v9 = _mm_loadu_si128(v8 + 5);
  v10 = v8->m128i_i64[0];
  v79 = v8->m128i_i64[1];
  v96 = v8[5].m128i_i64[0];
  v92 = v8[5].m128i_u32[2];
  v11 = sub_1E34390(*(_QWORD *)(a3 + 104));
  v12 = *(_QWORD *)(a3 + 104);
  v13 = *(_WORD *)(v12 + 32);
  v14 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v15 = *(_QWORD *)(v12 + 56);
  v116 = v14;
  v117 = v15;
  v101 = v13;
  v16 = sub_20363F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a3 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a3 + 32) + 48LL));
  v17 = *(_QWORD *)(a3 + 72);
  v85 = v16;
  v86 = v18;
  v108 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v108, v17, 2);
  v19 = *(_BYTE *)(a3 + 88);
  v109 = *(_DWORD *)(a3 + 64);
  v20 = *(_QWORD *)(a3 + 96);
  v110[0] = v19;
  v111 = v20;
  v21 = *(_QWORD *)(v85 + 40) + 16LL * (unsigned int)v86;
  v22 = *(_BYTE *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  v112[0] = v22;
  v113 = v23;
  LOBYTE(v24) = sub_1F7E0F0((__int64)v110);
  v83 = v24;
  v84 = v25;
  LOBYTE(v26) = sub_1F7E0F0((__int64)v112);
  v114 = v26;
  v115 = v27;
  if ( (_BYTE)v26 )
    v28 = sub_2021900(v26);
  else
    v28 = sub_1F58D40((__int64)&v114);
  v80 = v28 >> 3;
  if ( v110[0] )
    v81 = word_4305480[(unsigned __int8)(v110[0] - 14)];
  else
    v81 = sub_1F58D30((__int64)v110);
  v29 = (__int64 *)a1[1];
  v94 = *a1;
  v98 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
  v30 = sub_1E0A0C0(v29[4]);
  if ( v98 == sub_1D13A20 )
  {
    v31 = 8 * sub_15A9520(v30, 0);
    if ( v31 == 32 )
    {
      v32 = 5;
    }
    else if ( v31 > 0x20 )
    {
      v32 = 6;
      if ( v31 != 64 )
      {
        v32 = 0;
        if ( v31 == 128 )
          v32 = 7;
      }
    }
    else
    {
      v32 = 3;
      if ( v31 != 8 )
        v32 = 4 * (v31 == 16);
    }
  }
  else
  {
    v32 = v98(v94, v30);
  }
  *(_QWORD *)&v33 = sub_1D38BB0((__int64)v29, 0, (__int64)&v108, v32, 0, 0, v9, *(double *)v14.m128i_i64, a6, 0);
  v34 = sub_1D332F0(
          v29,
          106,
          (__int64)&v108,
          v114,
          v115,
          0,
          *(double *)v9.m128i_i64,
          *(double *)v14.m128i_i64,
          a6,
          v85,
          v86,
          v33);
  v78 = v101;
  v36 = sub_1D2C750(
          (_QWORD *)a1[1],
          v10,
          v79,
          (__int64)&v108,
          (__int64)v34,
          v35,
          v9.m128i_i64[0],
          v9.m128i_i64[1],
          *(_OWORD *)*(_QWORD *)(a3 + 104),
          *(_QWORD *)(*(_QWORD *)(a3 + 104) + 16LL),
          v83,
          v84,
          v11,
          v101,
          (__int64)&v116);
  v38 = v37;
  v39 = v36;
  v40 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v40 >= *(_DWORD *)(a2 + 12) )
  {
    v104 = v36;
    v107 = v38;
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v36, v38);
    v40 = *(unsigned int *)(a2 + 8);
    v39 = v104;
    v38 = v107;
  }
  v41 = (__int64 *)(*(_QWORD *)a2 + 16 * v40);
  *v41 = v39;
  v41[1] = v38;
  ++*(_DWORD *)(a2 + 8);
  if ( v81 > 1 )
  {
    v75 = v10;
    v97 = 1;
    v42 = v89;
    v76 = v92;
    v93 = 16LL * v92;
    v43 = v11;
    v44 = v87;
    v77 = v43;
    v95 = v80;
    do
    {
      v52 = (__int64 *)a1[1];
      v53 = *(_QWORD *)(v96 + 40) + v93;
      LOBYTE(v44) = *(_BYTE *)v53;
      v54 = v91;
      v55 = sub_1D38BB0(
              (__int64)v52,
              v95,
              (__int64)&v108,
              v44,
              *(const void ***)(v53 + 8),
              0,
              v9,
              *(double *)v14.m128i_i64,
              a6,
              0);
      v56 = *(_QWORD *)(v96 + 40) + v93;
      LOBYTE(v54) = *(_BYTE *)v56;
      *((_QWORD *)&v74 + 1) = v57;
      *(_QWORD *)&v74 = v55;
      v100 = v76 | v100 & 0xFFFFFFFF00000000LL;
      v102 = sub_1D332F0(
               v52,
               52,
               (__int64)&v108,
               v54,
               *(const void ***)(v56 + 8),
               3u,
               *(double *)v9.m128i_i64,
               *(double *)v14.m128i_i64,
               a6,
               v96,
               v100,
               v74);
      v58 = (__int64 *)a1[1];
      v105 = v59;
      v88 = *a1;
      v90 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
      v60 = sub_1E0A0C0(v58[4]);
      if ( v90 == sub_1D13A20 )
      {
        v61 = 8 * sub_15A9520(v60, 0);
        if ( v61 == 32 )
        {
          v62 = 5;
        }
        else if ( v61 > 0x20 )
        {
          v62 = 6;
          if ( v61 != 64 )
          {
            v62 = 0;
            if ( v61 == 128 )
              v62 = 7;
          }
        }
        else
        {
          v62 = 3;
          if ( v61 != 8 )
            v62 = 4 * (v61 == 16);
        }
      }
      else
      {
        v62 = v90(v88, v60);
      }
      LOBYTE(v42) = v62;
      *(_QWORD *)&v63 = sub_1D38BB0((__int64)v58, 0, (__int64)&v108, v42, 0, 0, v9, *(double *)v14.m128i_i64, a6, 0);
      v64 = sub_1D332F0(
              v58,
              106,
              (__int64)&v108,
              v114,
              v115,
              0,
              *(double *)v9.m128i_i64,
              *(double *)v14.m128i_i64,
              a6,
              v85,
              v86,
              v63);
      v66 = v65;
      v67 = *(_QWORD *)(a3 + 104);
      v68 = (_QWORD *)a1[1];
      v69 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v69 )
      {
        v70 = *(_QWORD *)(v67 + 8) + v95;
        v71 = *(_BYTE *)(v67 + 16);
        if ( (*(_QWORD *)v67 & 4) != 0 )
        {
          *((_QWORD *)&v118 + 1) = *(_QWORD *)(v67 + 8) + v95;
          LOBYTE(v119) = v71;
          *(_QWORD *)&v118 = v69 | 4;
          HIDWORD(v119) = *(_DWORD *)(v69 + 12);
        }
        else
        {
          v72 = *(_QWORD *)v69;
          *(_QWORD *)&v118 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v118 + 1) = v70;
          v73 = *(_BYTE *)(v72 + 8) == 16;
          LOBYTE(v119) = v71;
          if ( v73 )
            v72 = **(_QWORD **)(v72 + 16);
          HIDWORD(v119) = *(_DWORD *)(v72 + 8) >> 8;
        }
      }
      else
      {
        v45 = *(_DWORD *)(v67 + 20);
        LODWORD(v119) = 0;
        v118 = 0u;
        HIDWORD(v119) = v45;
      }
      v46 = sub_1D2C750(
              v68,
              v75,
              v79,
              (__int64)&v108,
              (__int64)v64,
              v66,
              (__int64)v102,
              v105,
              v118,
              v119,
              v83,
              v84,
              -(v77 | v95) & (v77 | v95),
              v78,
              (__int64)&v116);
      v48 = v47;
      v49 = v46;
      v50 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v50 >= *(_DWORD *)(a2 + 12) )
      {
        v103 = v46;
        v106 = v48;
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v46, v48);
        v50 = *(unsigned int *)(a2 + 8);
        v49 = v103;
        v48 = v106;
      }
      v51 = (__int64 *)(*(_QWORD *)a2 + 16 * v50);
      ++v97;
      *v51 = v49;
      v51[1] = v48;
      v95 += v80;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( v97 != v81 );
  }
  if ( v108 )
    sub_161E7C0((__int64)&v108, v108);
}
