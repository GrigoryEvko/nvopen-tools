// Function: sub_2139D90
// Address: 0x2139d90
//
__int64 *__fastcall sub_2139D90(__int64 *a1, unsigned __int64 a2, int a3, double a4, double a5, __m128i a6)
{
  const __m128i *v7; // rax
  __int64 v8; // rsi
  __m128 v9; // xmm0
  __int64 v10; // rcx
  unsigned __int32 v11; // ebx
  unsigned __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  bool v15; // zf
  char v16; // dl
  unsigned __int64 v17; // rax
  __int64 *v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rbx
  unsigned int v21; // edx
  __int64 *v22; // r12
  unsigned __int64 v23; // r13
  const void ***v24; // rax
  int v25; // edx
  __int64 v26; // r9
  unsigned __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // rbx
  __int128 v34; // rax
  __int64 *v35; // rax
  __int64 *v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // r10
  unsigned int v43; // edx
  unsigned __int64 v44; // r11
  unsigned int v45; // edx
  const __m128i *v46; // r9
  unsigned int v48; // eax
  __int64 v49; // r11
  __int128 v50; // rax
  __int64 *v51; // rbx
  __int16 *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rdx
  unsigned int v58; // edx
  __int64 *v59; // rax
  unsigned int v60; // edx
  __int128 v61; // [rsp-20h] [rbp-120h]
  __int128 v62; // [rsp-10h] [rbp-110h]
  __int128 v63; // [rsp-10h] [rbp-110h]
  __int64 v64; // [rsp+8h] [rbp-F8h]
  __int64 *v65; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v66; // [rsp+20h] [rbp-E0h]
  const void **v67; // [rsp+30h] [rbp-D0h]
  __int128 v68; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v69; // [rsp+40h] [rbp-C0h]
  __int16 *v70; // [rsp+40h] [rbp-C0h]
  __int64 v71; // [rsp+50h] [rbp-B0h]
  __int64 *v72; // [rsp+50h] [rbp-B0h]
  __int64 *v73; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v74; // [rsp+50h] [rbp-B0h]
  __int64 *v75; // [rsp+50h] [rbp-B0h]
  __int16 *v76; // [rsp+58h] [rbp-A8h]
  __int64 *v77; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v78; // [rsp+68h] [rbp-98h]
  const void **v79; // [rsp+70h] [rbp-90h]
  __int64 *v80; // [rsp+80h] [rbp-80h]
  __int64 v81; // [rsp+B0h] [rbp-50h] BYREF
  int v82; // [rsp+B8h] [rbp-48h]
  unsigned int v83; // [rsp+C0h] [rbp-40h] BYREF
  unsigned __int64 v84; // [rsp+C8h] [rbp-38h]

  if ( a3 == 1 )
    return sub_2128280(a1, a2, a4, a5, a6);
  v7 = *(const __m128i **)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = (__m128)_mm_loadu_si128(v7);
  v10 = v7->m128i_i64[0];
  v81 = v8;
  v11 = v7->m128i_u32[2];
  v12 = v7[2].m128i_u64[1];
  v13 = v7[3].m128i_i64[0];
  if ( v8 )
  {
    v71 = v10;
    sub_1623A60((__int64)&v81, v8, 2);
    v10 = v71;
  }
  v82 = *(_DWORD *)(a2 + 64);
  v14 = *(_QWORD *)(v10 + 40) + 16LL * v11;
  v15 = *(_WORD *)(a2 + 24) == 74;
  v16 = *(_BYTE *)v14;
  v17 = *(_QWORD *)(v14 + 8);
  LOBYTE(v83) = v16;
  v84 = v17;
  if ( v15 )
  {
    v59 = sub_2139100((__int64)a1, v9.m128_u64[0], v9.m128_i64[1], *(double *)v9.m128_u64, a5, a6);
    v20 = v60;
    v72 = v59;
    v22 = sub_2139100((__int64)a1, v12, v13, *(double *)v9.m128_u64, a5, a6);
  }
  else
  {
    v18 = sub_2139210((__int64)a1, v9.m128_u64[0], v9.m128_i64[1], (__m128i)v9, a5, a6);
    v20 = v19;
    v72 = v18;
    v22 = sub_2139210((__int64)a1, v12, v13, (__m128i)v9, a5, a6);
  }
  v23 = v21 | v13 & 0xFFFFFFFF00000000LL;
  v24 = (const void ***)sub_1D252B0(
                          a1[1],
                          *(unsigned __int8 *)(v72[5] + 16 * v20),
                          *(_QWORD *)(v72[5] + 16 * v20 + 8),
                          *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 24LL));
  *((_QWORD *)&v62 + 1) = v23;
  *(_QWORD *)&v62 = v22;
  v77 = sub_1D37440(
          (__int64 *)a1[1],
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v81,
          v24,
          v25,
          v26,
          *(double *)v9.m128_u64,
          a5,
          a6,
          __PAIR128__(v20 | v9.m128_u64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v72),
          v62);
  v31 = v27;
  v32 = (unsigned int)v27;
  if ( *(_WORD *)(a2 + 24) == 75 )
  {
    v75 = (__int64 *)a1[1];
    if ( (_BYTE)v83 )
    {
      v48 = sub_2127930(v83);
    }
    else
    {
      v48 = sub_1F58D40((__int64)&v83);
      v49 = 0;
    }
    v64 = v49;
    *(_QWORD *)&v50 = sub_1D38E70((__int64)v75, v48, (__int64)&v81, 0, (__m128i)v9, a5, a6);
    v51 = sub_1D332F0(
            v75,
            124,
            (__int64)&v81,
            *(unsigned __int8 *)(v77[5] + 16 * v32),
            *(const void ***)(v77[5] + 16 * v32 + 8),
            0,
            *(double *)v9.m128_u64,
            a5,
            a6,
            (__int64)v77,
            v31,
            v50);
    v70 = v52;
    v65 = (__int64 *)a1[1];
    *(_QWORD *)&v68 = sub_1D38BB0(
                        (__int64)v65,
                        0,
                        (__int64)&v81,
                        *(unsigned __int8 *)(v51[5] + 16LL * (unsigned int)v52),
                        *(const void ***)(v51[5] + 16LL * (unsigned int)v52 + 8),
                        0,
                        (__m128i)v9,
                        a5,
                        a6,
                        0);
    v53 = *(_QWORD *)(a2 + 40);
    *((_QWORD *)&v68 + 1) = v54;
    v79 = *(const void ***)(v53 + 24);
    v66 = *(unsigned __int8 *)(v53 + 16);
    v56 = sub_1D28D50(v65, 0x16u, v54, v66, (__int64)v79, v55);
    v42 = sub_1D3A900(v65, 0x89u, (__int64)&v81, v66, v79, 0, v9, a5, a6, (unsigned __int64)v51, v70, v68, v56, v57);
    v44 = v58 | v64 & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v33 = 16LL * (unsigned int)v27;
    v73 = (__int64 *)a1[1];
    *(_QWORD *)&v34 = sub_1D2EF30(v73, v83, v84, v28, v29, v30);
    v35 = sub_1D332F0(
            v73,
            148,
            (__int64)&v81,
            *(unsigned __int8 *)(v77[5] + v33),
            *(const void ***)(v77[5] + v33 + 8),
            0,
            *(double *)v9.m128_u64,
            a5,
            a6,
            (__int64)v77,
            v31,
            v34);
    v36 = (__int64 *)a1[1];
    v74 = (unsigned __int64)v35;
    v37 = *(_QWORD *)(a2 + 40);
    v76 = (__int16 *)v38;
    v67 = *(const void ***)(v37 + 24);
    v69 = *(unsigned __int8 *)(v37 + 16);
    v40 = sub_1D28D50(v36, 0x16u, v38, v69, (__int64)v67, v39);
    *((_QWORD *)&v61 + 1) = v31;
    *(_QWORD *)&v61 = v77;
    v42 = sub_1D3A900(v36, 0x89u, (__int64)&v81, v69, v67, 0, v9, a5, a6, v74, v76, v61, v40, v41);
    v44 = v43;
  }
  v78 = v44;
  *((_QWORD *)&v63 + 1) = 1;
  *(_QWORD *)&v63 = v77;
  v80 = sub_1D332F0(
          (__int64 *)a1[1],
          119,
          (__int64)&v81,
          *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 24LL),
          0,
          *(double *)v9.m128_u64,
          a5,
          a6,
          (__int64)v42,
          v44,
          v63);
  sub_2013400((__int64)a1, a2, 1, (__int64)v80, (__m128i *)(v45 | v78 & 0xFFFFFFFF00000000LL), v46);
  if ( v81 )
    sub_161E7C0((__int64)&v81, v81);
  return v77;
}
