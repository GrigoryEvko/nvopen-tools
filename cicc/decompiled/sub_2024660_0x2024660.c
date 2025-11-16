// Function: sub_2024660
// Address: 0x2024660
//
unsigned __int64 __fastcall sub_2024660(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        _DWORD *a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v9; // rax
  __m128 v10; // xmm0
  __m128i v11; // xmm1
  unsigned __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r12
  char v18; // al
  __int64 v19; // rdx
  unsigned int v20; // r12d
  __int64 v21; // rax
  char v22; // dl
  const void **v23; // rax
  unsigned int v24; // edx
  int v25; // eax
  _QWORD *v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  int v35; // eax
  _QWORD *v36; // rdi
  __int64 v37; // r13
  __int64 v38; // rax
  unsigned int v39; // edx
  _QWORD *v40; // rdi
  unsigned __int64 v41; // r12
  unsigned __int64 v42; // rdx
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // rax
  char v46; // di
  const void **v47; // rax
  unsigned int v48; // r10d
  const void ***v49; // rax
  __int128 v50; // rax
  __int64 *v51; // rax
  _QWORD *v52; // rdi
  unsigned int v53; // edx
  __int64 v54; // r8
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rsi
  unsigned int v59; // edx
  unsigned __int64 result; // rax
  __int64 v61; // rcx
  _QWORD *v62; // rax
  char *v63; // rax
  __int64 v64; // rsi
  char v65; // dl
  __int64 v66; // rax
  int v67; // edx
  unsigned int v68; // [rsp+28h] [rbp-118h]
  unsigned int v69; // [rsp+28h] [rbp-118h]
  __int64 v70; // [rsp+30h] [rbp-110h]
  __int64 v71; // [rsp+30h] [rbp-110h]
  unsigned int v72; // [rsp+30h] [rbp-110h]
  __int64 v73; // [rsp+38h] [rbp-108h]
  __int64 v74; // [rsp+40h] [rbp-100h]
  _QWORD *v75; // [rsp+40h] [rbp-100h]
  unsigned __int64 v76; // [rsp+48h] [rbp-F8h]
  __int64 v77; // [rsp+48h] [rbp-F8h]
  int v78; // [rsp+48h] [rbp-F8h]
  __int64 *v81; // [rsp+58h] [rbp-E8h]
  __int64 v82; // [rsp+60h] [rbp-E0h]
  __int64 v83; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v84; // [rsp+68h] [rbp-D8h]
  __int64 v85; // [rsp+B0h] [rbp-90h] BYREF
  int v86; // [rsp+B8h] [rbp-88h]
  unsigned int v87; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v88; // [rsp+C8h] [rbp-78h]
  __int64 v89; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v90; // [rsp+D8h] [rbp-68h]
  __int64 v91; // [rsp+E0h] [rbp-60h]
  unsigned __int64 v92; // [rsp+F0h] [rbp-50h] BYREF
  const void **v93; // [rsp+F8h] [rbp-48h]
  __int64 v94; // [rsp+100h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 32);
  v10 = (__m128)_mm_loadu_si128((const __m128i *)(v9 + 40));
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 80));
  v70 = *(_QWORD *)(v9 + 8);
  v12 = *(_QWORD *)v9;
  v76 = *(_QWORD *)v9;
  v13 = *(unsigned int *)(v9 + 8);
  v68 = *(_DWORD *)(v9 + 48);
  v14 = *(_QWORD *)(a2 + 72);
  v74 = *(_QWORD *)(v9 + 40);
  v82 = *(_QWORD *)(v9 + 80);
  v85 = v14;
  if ( v14 )
    sub_1623A60((__int64)&v85, v14, 2);
  v86 = *(_DWORD *)(a2 + 64);
  sub_2017DE0(a1, v76, v70, a3, a4);
  v17 = *(_QWORD *)(v12 + 40) + 16 * v13;
  v18 = *(_BYTE *)v17;
  v19 = *(_QWORD *)(v17 + 8);
  LOBYTE(v87) = v18;
  v88 = v19;
  if ( v18 )
    v20 = word_4305480[(unsigned __int8)(v18 - 14)];
  else
    v20 = sub_1F58D30((__int64)&v87);
  v21 = *(_QWORD *)(v74 + 40) + 16LL * v68;
  v22 = *(_BYTE *)v21;
  v23 = *(const void ***)(v21 + 8);
  LOBYTE(v92) = v22;
  v93 = v23;
  if ( v22 )
  {
    v24 = word_4305480[(unsigned __int8)(v22 - 14)];
    v25 = *(unsigned __int16 *)(v82 + 24);
    if ( v25 != 32 )
    {
LABEL_7:
      if ( v25 != 10 )
        goto LABEL_8;
    }
  }
  else
  {
    v24 = sub_1F58D30((__int64)&v92);
    v25 = *(unsigned __int16 *)(v82 + 24);
    if ( v25 != 32 )
      goto LABEL_7;
  }
  v61 = *(_QWORD *)(v82 + 88);
  v62 = *(_QWORD **)(v61 + 24);
  if ( *(_DWORD *)(v61 + 32) > 0x40u )
    v62 = (_QWORD *)*v62;
  if ( !(_DWORD)v62 && v20 >> 1 >= v24 )
  {
    v63 = *(char **)(a2 + 40);
    v64 = *(_QWORD *)(a1 + 8);
    v65 = *v63;
    v66 = *((_QWORD *)v63 + 1);
    LOBYTE(v89) = v65;
    v90 = v66;
    sub_1D19A30((__int64)&v92, v64, &v89);
    *a3 = (unsigned __int64)sub_1D3A900(
                              *(__int64 **)(a1 + 8),
                              0x6Cu,
                              (__int64)&v85,
                              v92,
                              v93,
                              0,
                              v10,
                              *(double *)v11.m128i_i64,
                              a7,
                              *a3,
                              (__int16 *)a3[1],
                              *(_OWORD *)&v10,
                              v11.m128i_i64[0],
                              v11.m128i_i64[1]);
    *((_DWORD *)a3 + 2) = v67;
    return sub_17CD270(&v85);
  }
LABEL_8:
  v26 = sub_1D29C20(*(_QWORD **)(a1 + 8), v87, v88, 1, v15, v16);
  v27 = *(_QWORD **)(a1 + 8);
  v90 = 0;
  v91 = 0;
  v69 = v28;
  v84 = v28;
  v75 = v26;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v89 = 0;
  v83 = (__int64)v26;
  v29 = sub_1D2BF40(v27, (__int64)(v27 + 11), 0, (__int64)&v85, v76, v70, (__int64)v26, v28, 0, 0, 0, 0, (__int64)&v92);
  v31 = v30;
  v32 = v29;
  v71 = sub_20BD400(*(_QWORD *)a1, *(_QWORD *)(a1 + 8), v83, v84, v87, v88, v11.m128i_i64[0], v11.m128i_i64[1]);
  v73 = v33;
  v77 = sub_1F58E60((__int64)&v87, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL));
  v34 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
  v35 = sub_15AAE50(v34, v77);
  v36 = *(_QWORD **)(a1 + 8);
  v90 = 0;
  v91 = 0;
  v78 = v35;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v89 = 0;
  v37 = sub_1D2BF40(v36, v32, v31, (__int64)&v85, v10.m128_i64[0], v10.m128_i64[1], v71, v73, 0, 0, 0, 0, (__int64)&v92);
  v38 = *((unsigned int *)a3 + 2);
  v90 = 0;
  v91 = 0;
  v40 = *(_QWORD **)(a1 + 8);
  v41 = v39 | v31 & 0xFFFFFFFF00000000LL;
  v42 = *a3;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v89 = 0;
  v43 = sub_1D2B730(
          v40,
          *(unsigned __int8 *)(*(_QWORD *)(v42 + 40) + 16 * v38),
          *(_QWORD *)(*(_QWORD *)(v42 + 40) + 16 * v38 + 8),
          (__int64)&v85,
          v37,
          v41,
          v83,
          v84,
          0,
          0,
          0,
          0,
          (__int64)&v92,
          0);
  *a3 = v43;
  *((_DWORD *)a3 + 2) = v44;
  v45 = *(_QWORD *)(v43 + 40) + 16LL * v44;
  v46 = *(_BYTE *)v45;
  v47 = *(const void ***)(v45 + 8);
  LOBYTE(v92) = v46;
  v93 = v47;
  if ( v46 )
    v48 = sub_2021900(v46);
  else
    v48 = sub_1F58D40((__int64)&v92);
  v72 = v48 >> 3;
  v81 = *(__int64 **)(a1 + 8);
  v49 = (const void ***)(v75[5] + 16LL * v69);
  *(_QWORD *)&v50 = sub_1D38BB0(
                      (__int64)v81,
                      v48 >> 3,
                      (__int64)&v85,
                      *(unsigned __int8 *)v49,
                      v49[1],
                      0,
                      (__m128i)v10,
                      *(double *)v11.m128i_i64,
                      a7,
                      0);
  v51 = sub_1D332F0(
          v81,
          52,
          (__int64)&v85,
          *(unsigned __int8 *)(v75[5] + 16LL * v69),
          *(const void ***)(v75[5] + 16LL * v69 + 8),
          0,
          *(double *)v10.m128_u64,
          *(double *)v11.m128i_i64,
          a7,
          v83,
          v84,
          v50);
  v52 = *(_QWORD **)(a1 + 8);
  v90 = 0;
  v91 = 0;
  v54 = v53;
  v55 = *(_QWORD *)a4;
  v56 = (unsigned int)a4[2];
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v89 = 0;
  v57 = sub_1D2B730(
          v52,
          *(unsigned __int8 *)(*(_QWORD *)(v55 + 40) + 16 * v56),
          *(_QWORD *)(*(_QWORD *)(v55 + 40) + 16 * v56 + 8),
          (__int64)&v85,
          v37,
          v41,
          (__int64)v51,
          v54 | v84 & 0xFFFFFFFF00000000LL,
          0,
          0,
          -(v78 | v72) & (v78 | v72),
          0,
          (__int64)&v92,
          0);
  v58 = v85;
  *(_QWORD *)a4 = v57;
  result = v59;
  a4[2] = v59;
  if ( v58 )
    return sub_161E7C0((__int64)&v85, v58);
  return result;
}
