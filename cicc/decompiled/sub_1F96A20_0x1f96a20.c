// Function: sub_1F96A20
// Address: 0x1f96a20
//
__int64 *__fastcall sub_1F96A20(int *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *result; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v10; // xmm0
  __int64 v11; // r14
  __int64 v12; // rax
  char v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  __int128 v17; // xmm1
  int v18; // r14d
  unsigned int v19; // eax
  unsigned int v20; // edx
  __int64 *v21; // rsi
  __m128i v22; // xmm2
  __int64 *v23; // rsi
  __m128i v24; // xmm3
  __int64 v25; // rax
  __int8 v26; // dl
  __int64 v27; // rax
  __m128i v28; // xmm5
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // r11
  int v32; // edx
  __int64 v33; // rcx
  int v34; // r9d
  __int64 v35; // r11
  __int64 v36; // rax
  __int64 v37; // r14
  unsigned int v38; // edx
  unsigned int v39; // edx
  __int64 v40; // rcx
  __int64 v41; // r12
  __int64 v42; // r9
  int v43; // edx
  __int64 v44; // rcx
  int v45; // r9d
  unsigned int v46; // edx
  unsigned __int64 v47; // rdi
  __int32 v48; // eax
  __int64 v49; // rax
  unsigned int v50; // edx
  __int8 v51; // r8
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // eax
  int v55; // eax
  __int128 v56; // [rsp-1C8h] [rbp-1C8h]
  __int64 v57; // [rsp-1B8h] [rbp-1B8h]
  int v58; // [rsp-1B0h] [rbp-1B0h]
  int v59; // [rsp-1A8h] [rbp-1A8h]
  __int64 v60; // [rsp-1A8h] [rbp-1A8h]
  __int64 v61; // [rsp-198h] [rbp-198h]
  __int64 v62; // [rsp-190h] [rbp-190h]
  __int64 v63; // [rsp-190h] [rbp-190h]
  __int64 v64; // [rsp-188h] [rbp-188h]
  int v65; // [rsp-17Ch] [rbp-17Ch]
  __int64 v66; // [rsp-178h] [rbp-178h]
  __int64 v67; // [rsp-170h] [rbp-170h]
  unsigned int v68; // [rsp-168h] [rbp-168h]
  __int64 v69; // [rsp-168h] [rbp-168h]
  unsigned __int64 v70; // [rsp-160h] [rbp-160h]
  __int128 v71; // [rsp-158h] [rbp-158h]
  __int128 v72; // [rsp-148h] [rbp-148h]
  __int64 *v73; // [rsp-138h] [rbp-138h]
  __int128 v74; // [rsp-138h] [rbp-138h]
  int v75; // [rsp-138h] [rbp-138h]
  __int64 v76; // [rsp-130h] [rbp-130h]
  __int64 v77; // [rsp-128h] [rbp-128h]
  __m128i v78; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v79; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v80; // [rsp-E0h] [rbp-E0h]
  __int64 v81; // [rsp-D8h] [rbp-D8h] BYREF
  int v82; // [rsp-D0h] [rbp-D0h]
  _QWORD v83[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v84; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v85; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v86; // [rsp-98h] [rbp-98h] BYREF
  __m128i v87; // [rsp-88h] [rbp-88h] BYREF
  __m128i v88; // [rsp-78h] [rbp-78h] BYREF
  __int64 v89; // [rsp-68h] [rbp-68h]
  __m128i v90; // [rsp-58h] [rbp-58h] BYREF
  __m128i v91[4]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v92; // [rsp-8h] [rbp-8h] BYREF

  if ( a1[4] > 0 )
    return 0;
  if ( *(_WORD *)(a2 + 24) != 236 )
    BUG();
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v10 = _mm_loadu_si128((const __m128i *)(v7 + 120));
  v11 = *(_QWORD *)(v7 + 80);
  v78 = v10;
  v12 = *(_QWORD *)(v10.m128i_i64[0] + 40) + 16LL * v10.m128i_u32[2];
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v81 = v8;
  LOBYTE(v79) = v13;
  v80 = v14;
  if ( v8 )
    sub_1623A60((__int64)&v81, v8, 2);
  v82 = *(_DWORD *)(a2 + 64);
  result = 0;
  if ( *(_WORD *)(v11 + 24) == 137 )
  {
    sub_1F40D10((__int64)&v90, *((_QWORD *)a1 + 1), *(_QWORD *)(*(_QWORD *)a1 + 48LL), v79, v80);
    if ( v90.m128i_i8[0] == 6 )
    {
      sub_1F6F630((__int64)&v90, v11, *(__int64 **)a1, *(double *)v10.m128i_i64, a4, a5);
      *(_QWORD *)&v74 = v90.m128i_i64[0];
      *((_QWORD *)&v74 + 1) = v90.m128i_u32[2];
      *(_QWORD *)&v71 = v91[0].m128i_i64[0];
      *((_QWORD *)&v71 + 1) = v91[0].m128i_u32[2];
      v15 = *(_QWORD *)(a2 + 32);
      v16 = *(_QWORD *)v15;
      v17 = (__int128)_mm_loadu_si128((const __m128i *)(v15 + 40));
      LOBYTE(v83[0]) = *(_BYTE *)(a2 + 88);
      v67 = v16;
      v66 = *(_QWORD *)(v15 + 8);
      v83[1] = *(_QWORD *)(a2 + 96);
      v68 = 1 << *(_WORD *)(*(_QWORD *)(a2 + 104) + 34LL);
      v18 = v68 >> 1;
      if ( v13 )
        v19 = sub_1F6C8D0(v13);
      else
        v19 = sub_1F58D40((__int64)&v79);
      v20 = v68 >> 2;
      v21 = *(__int64 **)a1;
      v84.m128i_i8[0] = 0;
      if ( v19 >> 3 != v18 )
        v20 = v68 >> 1;
      v84.m128i_i64[1] = 0;
      v85.m128i_i8[0] = 0;
      v65 = v20;
      v85.m128i_i64[1] = 0;
      sub_1D19A30((__int64)&v90, (__int64)v21, v83);
      v22 = _mm_loadu_si128(&v90);
      v23 = *(__int64 **)a1;
      v24 = _mm_loadu_si128(v91);
      v86.m128i_i64[1] = 0;
      v84 = v22;
      v85 = v24;
      v25 = *(_QWORD *)(v78.m128i_i64[0] + 40) + 16LL * v78.m128i_u32[2];
      v87.m128i_i64[1] = 0;
      v86.m128i_i8[0] = 0;
      v87.m128i_i8[0] = 0;
      v26 = *(_BYTE *)v25;
      v27 = *(_QWORD *)(v25 + 8);
      v88.m128i_i8[0] = v26;
      v88.m128i_i64[1] = v27;
      sub_1D19A30((__int64)&v90, (__int64)v23, &v88);
      v28 = _mm_loadu_si128(v91);
      v86 = _mm_loadu_si128(&v90);
      v87 = v28;
      sub_1D40600(
        (__int64)&v90,
        v23,
        (__int64)&v78,
        (__int64)&v81,
        (const void ***)&v86,
        (const void ***)&v87,
        v10,
        *(double *)&v17,
        v22);
      v29 = *(_QWORD *)(a2 + 104);
      v30 = *(_QWORD *)(v29 + 64);
      v62 = v90.m128i_i64[0];
      v69 = v90.m128i_u32[2];
      v61 = v91[0].m128i_i64[0];
      v64 = v91[0].m128i_u32[2];
      v31 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      v90 = _mm_loadu_si128((const __m128i *)(v29 + 40));
      v91[0].m128i_i64[0] = *(_QWORD *)(v29 + 56);
      if ( v84.m128i_i8[0] )
      {
        v32 = sub_1F6C8D0(v84.m128i_i8[0]);
      }
      else
      {
        v57 = v31;
        v58 = v30;
        v60 = v29;
        v55 = sub_1F58D40((__int64)&v84);
        v35 = v57;
        v34 = v58;
        v33 = v60;
        v32 = v55;
      }
      v36 = sub_1E0B8E0(
              v35,
              2u,
              (unsigned int)(v32 + 7) >> 3,
              v18,
              (unsigned int)&v92 - 80,
              v34,
              *(_OWORD *)v33,
              *(_QWORD *)(v33 + 16),
              1u,
              0,
              0);
      v37 = sub_1D2C870(
              *(_QWORD **)a1,
              v67,
              v66,
              (__int64)&v81,
              v62,
              v69,
              v17,
              v74,
              v84.m128i_i64[0],
              v84.m128i_i64[1],
              v36,
              (*(_BYTE *)(a2 + 27) & 4) != 0,
              (*(_BYTE *)(a2 + 27) & 8) != 0);
      v70 = v38;
      *(_QWORD *)&v72 = sub_20BCE60(
                          *((_QWORD *)a1 + 1),
                          v17,
                          DWORD2(v17),
                          v74,
                          DWORD2(v74),
                          (unsigned int)&v92 - 208,
                          v84.m128i_i8[0],
                          v84.m128i_i64[1],
                          *(_QWORD *)a1,
                          (*(_BYTE *)(a2 + 27) & 8) != 0);
      *((_QWORD *)&v72 + 1) = v39 | *((_QWORD *)&v17 + 1) & 0xFFFFFFFF00000000LL;
      if ( v84.m128i_i8[0] )
        v75 = sub_1F6C8D0(v84.m128i_i8[0]);
      else
        v75 = sub_1F58D40((__int64)&v84);
      v40 = *(_QWORD *)(a2 + 104);
      v41 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      v42 = *(_QWORD *)(v40 + 64);
      v88 = _mm_loadu_si128((const __m128i *)(v40 + 40));
      v89 = *(_QWORD *)(v40 + 56);
      if ( v85.m128i_i8[0] )
      {
        v43 = sub_1F6C8D0(v85.m128i_i8[0]);
      }
      else
      {
        v59 = v42;
        v63 = v40;
        v54 = sub_1F58D40((__int64)&v85);
        v45 = v59;
        v44 = v63;
        v43 = v54;
      }
      v46 = (unsigned int)(v43 + 7) >> 3;
      v47 = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v47 )
      {
        v51 = *(_BYTE *)(v44 + 16);
        v52 = *(_QWORD *)(v44 + 8) + ((unsigned int)(v75 + 7) >> 3);
        if ( (*(_QWORD *)v44 & 4) != 0 )
        {
          v90.m128i_i64[1] = *(_QWORD *)(v44 + 8) + ((unsigned int)(v75 + 7) >> 3);
          v91[0].m128i_i8[0] = v51;
          v90.m128i_i64[0] = v47 | 4;
          v91[0].m128i_i32[1] = *(_DWORD *)(v47 + 12);
        }
        else
        {
          v90.m128i_i64[0] = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL;
          v90.m128i_i64[1] = v52;
          v91[0].m128i_i8[0] = v51;
          v53 = *(_QWORD *)v47;
          if ( *(_BYTE *)(*(_QWORD *)v47 + 8LL) == 16 )
            v53 = **(_QWORD **)(v53 + 16);
          v91[0].m128i_i32[1] = *(_DWORD *)(v53 + 8) >> 8;
        }
      }
      else
      {
        v48 = *(_DWORD *)(v44 + 20);
        v91[0].m128i_i32[0] = 0;
        v90 = 0u;
        v91[0].m128i_i32[1] = v48;
      }
      v49 = sub_1E0B8E0(v41, 2u, v46, v65, (int)&v88, v45, *(_OWORD *)&v90, v91[0].m128i_i64[0], 1u, 0, 0);
      v77 = sub_1D2C870(
              *(_QWORD **)a1,
              v67,
              v66,
              (__int64)&v81,
              v61,
              v64,
              v72,
              v71,
              v85.m128i_i64[0],
              v85.m128i_i64[1],
              v49,
              (*(_BYTE *)(a2 + 27) & 4) != 0,
              (*(_BYTE *)(a2 + 27) & 8) != 0);
      v76 = v50;
      sub_1F81BC0((__int64)a1, v37);
      sub_1F81BC0((__int64)a1, v77);
      *((_QWORD *)&v56 + 1) = v76;
      *(_QWORD *)&v56 = v77;
      result = sub_1D332F0(
                 *(__int64 **)a1,
                 2,
                 (__int64)&v81,
                 1,
                 0,
                 0,
                 *(double *)v10.m128i_i64,
                 *(double *)&v17,
                 v22,
                 v37,
                 v70,
                 v56);
    }
    else
    {
      result = 0;
    }
  }
  if ( v81 )
  {
    v73 = result;
    sub_161E7C0((__int64)&v81, v81);
    return v73;
  }
  return result;
}
