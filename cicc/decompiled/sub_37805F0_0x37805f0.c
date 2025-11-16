// Function: sub_37805F0
// Address: 0x37805f0
//
void __fastcall sub_37805F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int16 *v8; // rax
  __int16 v9; // dx
  __m128i v10; // xmm0
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  __int64 v13; // rdi
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rsi
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  __int64 v19; // rax
  unsigned __int16 *v20; // rax
  _QWORD *v21; // rsi
  __int64 v22; // rax
  __int16 v23; // dx
  __int64 v24; // rax
  __m128i v25; // xmm4
  unsigned __int16 v26; // dx
  __int64 v27; // rax
  __int64 v28; // rsi
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  unsigned __int16 *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // r8
  __int64 v35; // rdx
  _QWORD *v36; // rsi
  __int64 v37; // rax
  __int16 v38; // dx
  __int64 v39; // rax
  __m128i v40; // xmm2
  _QWORD *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // r9
  __m128i v44; // xmm0
  const __m128i *v45; // rax
  __m128i *v46; // rax
  bool v47; // zf
  int v48; // edx
  unsigned int v49; // edx
  unsigned __int8 *v50; // rax
  unsigned int v51; // edx
  __m128i v52; // rax
  __int64 v53; // r8
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rcx
  int v56; // edx
  char v57; // si
  int v58; // eax
  __int64 v59; // r9
  _QWORD *v60; // rdi
  const __m128i *v61; // rax
  int v62; // edx
  __int64 v63; // rdi
  __int128 v64; // [rsp-20h] [rbp-210h]
  __int128 v65; // [rsp-10h] [rbp-200h]
  __int64 v66; // [rsp+8h] [rbp-1E8h]
  __int64 v67; // [rsp+20h] [rbp-1D0h]
  __int64 v68; // [rsp+28h] [rbp-1C8h]
  __int128 v69; // [rsp+30h] [rbp-1C0h]
  unsigned __int8 v70; // [rsp+43h] [rbp-1ADh]
  char v71; // [rsp+44h] [rbp-1ACh]
  __int64 v74; // [rsp+58h] [rbp-198h]
  unsigned __int64 v75; // [rsp+60h] [rbp-190h]
  unsigned __int64 v76; // [rsp+68h] [rbp-188h]
  unsigned __int8 *v77; // [rsp+70h] [rbp-180h]
  char v78; // [rsp+BFh] [rbp-131h] BYREF
  __m128i v79; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v80; // [rsp+D0h] [rbp-120h] BYREF
  int v81; // [rsp+D8h] [rbp-118h]
  __m128i v82; // [rsp+E0h] [rbp-110h] BYREF
  __m128i v83; // [rsp+F0h] [rbp-100h] BYREF
  __int128 v84; // [rsp+100h] [rbp-F0h] BYREF
  __int128 v85; // [rsp+110h] [rbp-E0h] BYREF
  unsigned __int16 v86; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v87; // [rsp+128h] [rbp-C8h]
  __m128i v88; // [rsp+130h] [rbp-C0h] BYREF
  __int128 v89; // [rsp+140h] [rbp-B0h] BYREF
  __int128 v90; // [rsp+150h] [rbp-A0h] BYREF
  __m128i v91; // [rsp+160h] [rbp-90h] BYREF
  __m128i v92; // [rsp+170h] [rbp-80h] BYREF
  __int128 v93; // [rsp+180h] [rbp-70h] BYREF
  __int64 v94; // [rsp+190h] [rbp-60h]
  __m128i v95; // [rsp+1A0h] [rbp-50h] BYREF
  __m128i v96; // [rsp+1B0h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v79.m128i_i16[0] = 0;
  v79.m128i_i64[1] = 0;
  v80 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v80, v6, 1);
  v7 = *(_QWORD *)(a1 + 8);
  v81 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  *((_QWORD *)&v93 + 1) = *((_QWORD *)v8 + 1);
  LOWORD(v93) = v9;
  sub_33D0340((__int64)&v95, v7, (__int64 *)&v93);
  v10 = _mm_loadu_si128(&v95);
  v67 = v96.m128i_i64[0];
  v79 = v10;
  v66 = v96.m128i_i64[1];
  v11 = *(_QWORD *)(a2 + 40);
  v12 = *(_QWORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  *(_QWORD *)&v84 = 0;
  v14 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v15 = _mm_loadu_si128((const __m128i *)(v11 + 80));
  DWORD2(v84) = 0;
  v16 = *(_QWORD *)(v11 + 120);
  v17 = _mm_loadu_si128((const __m128i *)(v11 + 120));
  v74 = v13;
  v18 = _mm_loadu_si128((const __m128i *)(v11 + 160));
  v19 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)&v85 = 0;
  DWORD2(v85) = 0;
  v69 = (__int128)v15;
  v70 = *(_BYTE *)(v19 + 34);
  LOBYTE(v19) = *(_BYTE *)(a2 + 33);
  v82 = v17;
  v83 = v18;
  v71 = ((unsigned __int8)v19 >> 2) & 3;
  if ( *(_DWORD *)(v16 + 24) == 208 )
  {
    sub_377EF80((__int64 *)a1, v16, (__int64)&v84, (__int64)&v85, v10);
  }
  else
  {
    v20 = (unsigned __int16 *)(*(_QWORD *)(v16 + 48) + 16LL * v82.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v95, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v20, *((_QWORD *)v20 + 1));
    if ( v95.m128i_i8[0] == 6 )
    {
      sub_375E8D0(a1, v82.m128i_u64[0], v82.m128i_i64[1], (__int64)&v84, (__int64)&v85);
    }
    else
    {
      v21 = *(_QWORD **)(a1 + 8);
      v92.m128i_i16[0] = 0;
      v91.m128i_i16[0] = 0;
      v91.m128i_i64[1] = 0;
      v92.m128i_i64[1] = 0;
      v22 = *(_QWORD *)(v82.m128i_i64[0] + 48) + 16LL * v82.m128i_u32[2];
      v23 = *(_WORD *)v22;
      v24 = *(_QWORD *)(v22 + 8);
      LOWORD(v93) = v23;
      *((_QWORD *)&v93 + 1) = v24;
      sub_33D0340((__int64)&v95, (__int64)v21, (__int64 *)&v93);
      v25 = _mm_loadu_si128(&v96);
      v91 = _mm_loadu_si128(&v95);
      v92 = v25;
      sub_3408290(
        (__int64)&v95,
        v21,
        (__int128 *)v82.m128i_i8,
        (__int64)&v80,
        (unsigned int *)&v91,
        (unsigned int *)&v92,
        v10);
      *(_QWORD *)&v84 = v95.m128i_i64[0];
      DWORD2(v84) = v95.m128i_i32[2];
      *(_QWORD *)&v85 = v96.m128i_i64[0];
      DWORD2(v85) = v96.m128i_i32[2];
    }
  }
  v26 = *(_WORD *)(a2 + 96);
  v27 = *(_QWORD *)(a2 + 104);
  v28 = *(_QWORD *)(a1 + 8);
  v88.m128i_i16[0] = 0;
  v86 = v26;
  v87 = v27;
  v88.m128i_i64[1] = 0;
  v78 = 0;
  sub_33D04E0((__int64)&v95, v28, &v86, (unsigned __int16 *)&v79, &v78);
  v29 = _mm_loadu_si128(&v95);
  DWORD2(v89) = 0;
  v30 = _mm_loadu_si128(&v96);
  DWORD2(v90) = 0;
  v31 = (unsigned __int16 *)(*(_QWORD *)(v83.m128i_i64[0] + 48) + 16LL * v83.m128i_u32[2]);
  v32 = *(_QWORD *)(a1 + 8);
  v88 = v29;
  v33 = *(_QWORD *)a1;
  v34 = *((_QWORD *)v31 + 1);
  *(_QWORD *)&v89 = 0;
  v35 = *(_QWORD *)(v32 + 64);
  *(_QWORD *)&v90 = 0;
  sub_2FE6CC0((__int64)&v95, v33, v35, *v31, v34);
  if ( v95.m128i_i8[0] == 6 )
  {
    sub_375E8D0(a1, v83.m128i_u64[0], v83.m128i_i64[1], (__int64)&v89, (__int64)&v90);
  }
  else
  {
    v92.m128i_i64[1] = 0;
    v91.m128i_i16[0] = 0;
    v92.m128i_i16[0] = 0;
    v91.m128i_i64[1] = 0;
    v36 = *(_QWORD **)(a1 + 8);
    v37 = *(_QWORD *)(v83.m128i_i64[0] + 48) + 16LL * v83.m128i_u32[2];
    v38 = *(_WORD *)v37;
    v39 = *(_QWORD *)(v37 + 8);
    LOWORD(v93) = v38;
    *((_QWORD *)&v93 + 1) = v39;
    sub_33D0340((__int64)&v95, (__int64)v36, (__int64 *)&v93);
    v40 = _mm_loadu_si128(&v96);
    v91 = _mm_loadu_si128(&v95);
    v92 = v40;
    sub_3408290(
      (__int64)&v95,
      v36,
      (__int128 *)v83.m128i_i8,
      (__int64)&v80,
      (unsigned int *)&v91,
      (unsigned int *)&v92,
      v10);
    *(_QWORD *)&v89 = v95.m128i_i64[0];
    DWORD2(v89) = v95.m128i_i32[2];
    *(_QWORD *)&v90 = v96.m128i_i64[0];
    DWORD2(v90) = v96.m128i_i32[2];
  }
  v41 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v42 = *(_QWORD *)(a2 + 112);
  v43 = *(_QWORD *)(v42 + 72);
  v95 = _mm_loadu_si128((const __m128i *)(v42 + 40));
  v44 = _mm_loadu_si128((const __m128i *)(v42 + 56));
  v96 = v44;
  v45 = (const __m128i *)sub_2E7BD70(v41, 1u, -1, v70, (int)&v95, v43, *(_OWORD *)v42, *(_QWORD *)(v42 + 16), 1u, 0, 0);
  v46 = sub_33E8F60(
          *(__int64 **)(a1 + 8),
          v79.m128i_u32[0],
          v79.m128i_i64[1],
          (__int64)&v80,
          v12,
          v74,
          v14.m128i_u64[0],
          v14.m128i_u64[1],
          v69,
          v84,
          v89,
          v88.m128i_i64[0],
          v88.m128i_i64[1],
          v45,
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          v71,
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  v47 = v78 == 0;
  *(_QWORD *)a3 = v46;
  *(_DWORD *)(a3 + 8) = v48;
  if ( !v47 )
  {
    *(_QWORD *)a4 = v46;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a3 + 8);
    goto LABEL_10;
  }
  v50 = sub_3465590(
          v44,
          *(_QWORD *)a1,
          v14.m128i_i64[0],
          v14.m128i_i64[1],
          v84,
          DWORD2(v84),
          (__int64)&v80,
          v88.m128i_u16[0],
          v88.m128i_i64[1],
          *(_QWORD *)(a1 + 8),
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  LODWORD(v94) = 0;
  v75 = (unsigned __int64)v50;
  v93 = 0u;
  v76 = v51 | v14.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  BYTE4(v94) = 0;
  if ( v88.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v88.m128i_i16[0] - 176) > 0x34u )
    {
LABEL_18:
      v68 = *(_QWORD *)(a2 + 112);
      v52.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v88);
      v53 = v68;
      v95 = v52;
      v54 = *(_QWORD *)(v68 + 8) + ((unsigned __int64)(v52.m128i_i64[0] + 7) >> 3);
      v55 = *(_QWORD *)v68 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v55 )
      {
        v57 = *(_BYTE *)(v68 + 20);
        if ( (*(_QWORD *)v68 & 4) != 0 )
        {
          v56 = *(_DWORD *)(v55 + 12);
          v55 |= 4u;
        }
        else
        {
          v63 = *(_QWORD *)(v55 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v63 + 8) - 17 <= 1 )
            v63 = **(_QWORD **)(v63 + 16);
          v56 = *(_DWORD *)(v63 + 8) >> 8;
        }
      }
      else
      {
        v56 = *(_DWORD *)(v68 + 16);
        v57 = 0;
      }
      *(_QWORD *)&v93 = v55;
      *((_QWORD *)&v93 + 1) = v54;
      LODWORD(v94) = v56;
      BYTE4(v94) = v57;
      goto LABEL_23;
    }
  }
  else if ( !sub_3007100((__int64)&v88) )
  {
    goto LABEL_18;
  }
  v58 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
  v53 = *(_QWORD *)(a2 + 112);
  LODWORD(v94) = v58;
LABEL_23:
  v59 = *(_QWORD *)(v53 + 72);
  v60 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v95 = _mm_loadu_si128((const __m128i *)(v53 + 40));
  v96 = _mm_loadu_si128((const __m128i *)(v53 + 56));
  v61 = (const __m128i *)sub_2E7BD70(v60, 1u, -1, v70, (int)&v95, v59, v93, v94, 1u, 0, 0);
  *(_QWORD *)a4 = sub_33E8F60(
                    *(__int64 **)(a1 + 8),
                    v67,
                    v66,
                    (__int64)&v80,
                    v12,
                    v74,
                    v75,
                    v76,
                    v69,
                    v85,
                    v90,
                    v30.m128i_i64[0],
                    v30.m128i_i64[1],
                    v61,
                    (*(_WORD *)(a2 + 32) >> 7) & 7,
                    v71,
                    (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  *(_DWORD *)(a4 + 8) = v62;
LABEL_10:
  *((_QWORD *)&v65 + 1) = 1;
  *(_QWORD *)&v65 = *(_QWORD *)a4;
  *((_QWORD *)&v64 + 1) = 1;
  *(_QWORD *)&v64 = *(_QWORD *)a3;
  v77 = sub_3406EB0(*(_QWORD **)(a1 + 8), 2u, (__int64)&v80, 1, 0, *(_QWORD *)(a1 + 8), v64, v65);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v77, v74 & 0xFFFFFFFF00000000LL | v49);
  if ( v80 )
    sub_B91220((__int64)&v80, v80);
}
