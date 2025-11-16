// Function: sub_1FEE680
// Address: 0x1fee680
//
__int64 __fastcall sub_1FEE680(
        __int64 a1,
        unsigned int a2,
        const void **a3,
        __m128i *a4,
        __int64 *a5,
        _DWORD *a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        _BYTE *a10,
        __int64 a11)
{
  unsigned __int8 v16; // si
  __int64 v17; // rdi
  unsigned int v18; // r9d
  signed int v20; // eax
  __m128i v21; // xmm0
  int v22; // edx
  signed int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rcx
  unsigned int v27; // edi
  int v28; // eax
  signed int v29; // eax
  int v30; // eax
  __int64 *v31; // rdi
  int v32; // r10d
  int v33; // r11d
  unsigned int v34; // esi
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // rax
  unsigned int v41; // r10d
  unsigned int v42; // edx
  __int32 v43; // edx
  char v44; // r15
  int v45; // edx
  __m128i v46; // xmm0
  int v47; // eax
  int v48; // r11d
  int v49; // r10d
  __int64 *v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 *v53; // rax
  unsigned int v54; // edx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 *v58; // rax
  unsigned int v59; // edx
  unsigned int v60; // [rsp+Ch] [rbp-E4h]
  unsigned int v61; // [rsp+10h] [rbp-E0h]
  int v62; // [rsp+10h] [rbp-E0h]
  __int128 v63; // [rsp+10h] [rbp-E0h]
  __int128 v64; // [rsp+10h] [rbp-E0h]
  unsigned int v65; // [rsp+20h] [rbp-D0h]
  unsigned int v66; // [rsp+20h] [rbp-D0h]
  __int128 v67; // [rsp+20h] [rbp-D0h]
  __int128 v68; // [rsp+20h] [rbp-D0h]
  __int64 *v69; // [rsp+20h] [rbp-D0h]
  unsigned int v70; // [rsp+30h] [rbp-C0h]
  __int64 *v71; // [rsp+30h] [rbp-C0h]
  __int128 v72; // [rsp+40h] [rbp-B0h]
  __int64 v73; // [rsp+48h] [rbp-A8h]
  __int64 *v74; // [rsp+50h] [rbp-A0h]
  __int64 v75; // [rsp+58h] [rbp-98h]
  unsigned __int64 v76; // [rsp+58h] [rbp-98h]
  __int64 v77; // [rsp+60h] [rbp-90h]
  unsigned int v78; // [rsp+60h] [rbp-90h]
  int v79; // [rsp+60h] [rbp-90h]

  v16 = *(_BYTE *)(*(_QWORD *)(a4->m128i_i64[0] + 40) + 16LL * a4->m128i_u32[2]);
  v17 = *(int *)(*(_QWORD *)a6 + 84LL);
  *a10 = 0;
  v18 = 0;
  if ( ((*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * ((v16 >> 3) + 15 * v17 + 18112) + 12) >> (4 * (v16 & 7))) & 0xF) != 0 )
  {
    v77 = v16 >> 3;
    v65 = 4 * (v16 & 7);
    v20 = sub_1D16ED0(v17);
    if ( ((*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * (v77 + 15LL * v20 + 18112) + 12) >> v65) & 0xB) == 0 )
    {
      v21 = _mm_loadu_si128(a4);
      a4->m128i_i64[0] = *a5;
      a4->m128i_i32[2] = *((_DWORD *)a5 + 2);
      *a5 = v21.m128i_i64[0];
      *((_DWORD *)a5 + 2) = v21.m128i_i32[2];
      *(_QWORD *)a6 = sub_1D28D50(*(_QWORD **)(a1 + 16), v20, (unsigned int)v17, v65, v77, v20);
      a6[2] = v22;
      return 1;
    }
    v70 = 4 * (v16 & 7);
    v61 = v17;
    v23 = sub_1D16EF0(v17, ((unsigned __int8)(v16 - 2) <= 5u) | (unsigned __int8)((unsigned __int8)(v16 - 14) <= 0x47u));
    v25 = *(_QWORD *)(a1 + 8);
    v26 = v65;
    v27 = v23;
    v28 = (*(_DWORD *)(v25 + 4 * (v77 + 15LL * v23 + 18112) + 12) >> v65) & 0xF;
    if ( !v28 || (v24 = v61, (_BYTE)v28 == 4) )
    {
      v44 = 0;
    }
    else
    {
      v66 = v61;
      v29 = sub_1D16ED0(v27);
      v25 = *(_QWORD *)(a1 + 8);
      v26 = v70;
      v24 = v61;
      v27 = v29;
      v30 = (*(_DWORD *)(v25 + 4 * (v77 + 15LL * v29 + 18112) + 12) >> v70) & 0xF;
      if ( v30 && (_BYTE)v30 != 4 )
      {
        if ( v61 != 7 )
        {
          if ( v61 <= 7 )
          {
            v48 = 7;
            v49 = 118;
          }
          else
          {
            if ( v61 == 8 )
            {
              v31 = *(__int64 **)(a1 + 16);
              v32 = 119;
              v33 = 14;
              v34 = 14;
LABEL_14:
              v62 = v32;
              v78 = v33;
              v67 = (__int128)*a4;
              v35 = sub_1D28D50(v31, v34, v24, v70, a4->m128i_i64[0], a4->m128i_i64[1]);
              v74 = sub_1D3A900(
                      v31,
                      0x89u,
                      a11,
                      a2,
                      a3,
                      0,
                      a7,
                      *(double *)a8.m128i_i64,
                      a9,
                      v67,
                      *((__int16 **)&v67 + 1),
                      v67,
                      v35,
                      v36);
              v68 = *(_OWORD *)a5;
              v71 = *(__int64 **)(a1 + 16);
              v76 = v37 | v75 & 0xFFFFFFFF00000000LL;
              v38 = sub_1D28D50(v71, v78, v37, 0xFFFFFFFF00000000LL, *a5, a5[1]);
              v40 = sub_1D3A900(
                      v71,
                      0x89u,
                      a11,
                      a2,
                      a3,
                      0,
                      a7,
                      *(double *)a8.m128i_i64,
                      a9,
                      v68,
                      *((__int16 **)&v68 + 1),
                      v68,
                      v38,
                      v39);
              v41 = v62;
              *(_QWORD *)&v72 = v40;
              *((_QWORD *)&v72 + 1) = v42 | v73 & 0xFFFFFFFF00000000LL;
LABEL_15:
              a4->m128i_i64[0] = (__int64)sub_1D332F0(
                                            *(__int64 **)(a1 + 16),
                                            v41,
                                            a11,
                                            a2,
                                            a3,
                                            0,
                                            *(double *)a7.m128_u64,
                                            *(double *)a8.m128i_i64,
                                            a9,
                                            (__int64)v74,
                                            v76,
                                            v72);
              a4->m128i_i32[2] = v43;
              *a5 = 0;
              *((_DWORD *)a5 + 2) = 0;
              *(_QWORD *)a6 = 0;
              a6[2] = 0;
              return 1;
            }
            v47 = v61 & 8;
            v48 = 8 - (v47 == 0);
            v49 = 119 - (v47 == 0);
          }
          v50 = *(__int64 **)(a1 + 16);
          a8 = _mm_loadu_si128(a4);
          v79 = v49;
          v60 = v48;
          v63 = *(_OWORD *)a5;
          v51 = sub_1D28D50(v50, v66 & 7 | 0x10, v66, 0, *a5, a5[1]);
          v53 = sub_1D3A900(
                  v50,
                  0x89u,
                  a11,
                  a2,
                  a3,
                  0,
                  a7,
                  *(double *)a8.m128i_i64,
                  a9,
                  a8.m128i_u64[0],
                  (__int16 *)a8.m128i_i64[1],
                  v63,
                  v51,
                  v52);
          a9 = _mm_loadu_si128(a4);
          v74 = v53;
          v64 = *(_OWORD *)a5;
          v69 = *(__int64 **)(a1 + 16);
          v76 = v54;
          v56 = sub_1D28D50(v69, v60, v54, v55, *a5, a5[1]);
          v58 = sub_1D3A900(
                  v69,
                  0x89u,
                  a11,
                  a2,
                  a3,
                  0,
                  a7,
                  *(double *)a8.m128i_i64,
                  a9,
                  a9.m128i_u64[0],
                  (__int16 *)a9.m128i_i64[1],
                  v64,
                  v56,
                  v57);
          v41 = v79;
          *(_QWORD *)&v72 = v58;
          *((_QWORD *)&v72 + 1) = v59;
          goto LABEL_15;
        }
        v31 = *(__int64 **)(a1 + 16);
        v32 = 118;
        v33 = 1;
        v34 = 1;
        goto LABEL_14;
      }
      v44 = 1;
    }
    *(_QWORD *)a6 = sub_1D28D50(*(_QWORD **)(a1 + 16), v27, v24, v26, *(_QWORD *)(a1 + 16), v25);
    a6[2] = v45;
    *a10 = 1;
    if ( v44 )
    {
      v46 = _mm_loadu_si128(a4);
      a4->m128i_i64[0] = *a5;
      a4->m128i_i32[2] = *((_DWORD *)a5 + 2);
      *a5 = v46.m128i_i64[0];
      *((_DWORD *)a5 + 2) = v46.m128i_i32[2];
    }
    return 1;
  }
  return v18;
}
