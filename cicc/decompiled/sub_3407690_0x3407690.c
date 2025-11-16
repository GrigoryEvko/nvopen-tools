// Function: sub_3407690
// Address: 0x3407690
//
__m128i *__fastcall sub_3407690(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __m128i v5; // xmm0
  __int16 *v6; // rdx
  __int16 v7; // cx
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  char v12; // r14
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // edx
  unsigned __int16 v17; // ax
  __int64 v18; // rdx
  __m128i *v19; // r11
  __int64 v20; // rax
  __m128i *v21; // r10
  __int64 v22; // rbx
  unsigned __int16 *v23; // rcx
  unsigned int v24; // r15d
  __int64 v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r9
  unsigned int v35; // edx
  __int64 v36; // r8
  unsigned __int8 *v37; // r15
  int v38; // eax
  __int64 v39; // r9
  unsigned __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int16 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rsi
  unsigned __int8 v45; // al
  __m128i *v46; // rax
  unsigned int v47; // edx
  __m128i *v48; // r14
  __int64 v50; // rbx
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned int v53; // edx
  __int64 v54; // rbx
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned __int8 *v57; // rax
  unsigned int v58; // edx
  __int128 v59; // [rsp-30h] [rbp-150h]
  unsigned __int64 v60; // [rsp+0h] [rbp-120h]
  __int64 v61; // [rsp+8h] [rbp-118h]
  __int64 v62; // [rsp+10h] [rbp-110h]
  __int64 v63; // [rsp+18h] [rbp-108h]
  unsigned __int8 v64; // [rsp+27h] [rbp-F9h]
  char v65; // [rsp+27h] [rbp-F9h]
  _BYTE *v66; // [rsp+30h] [rbp-F0h]
  __int64 v67; // [rsp+30h] [rbp-F0h]
  __m128i *v68; // [rsp+30h] [rbp-F0h]
  __int64 v69; // [rsp+50h] [rbp-D0h]
  __int64 v70; // [rsp+58h] [rbp-C8h]
  __int64 (__fastcall *v71)(__int64, __int64, unsigned int); // [rsp+60h] [rbp-C0h]
  __int128 v72; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v73; // [rsp+68h] [rbp-B8h]
  __int64 v74; // [rsp+90h] [rbp-90h] BYREF
  int v75; // [rsp+98h] [rbp-88h]
  unsigned int v76; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-78h]
  __int128 v78; // [rsp+B0h] [rbp-70h]
  __int64 v79; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v80; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v81; // [rsp+D8h] [rbp-48h]
  __int64 v82; // [rsp+E0h] [rbp-40h]
  __int64 v83; // [rsp+E8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v74 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v74, v3, 1);
  v75 = *(_DWORD *)(a2 + 72);
  v66 = *(_BYTE **)(a1 + 16);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v70 = *(_QWORD *)(*(_QWORD *)(v4 + 80) + 96LL);
  v77 = *((_QWORD *)v6 + 1);
  v8 = *(_QWORD *)(v4 + 8);
  LOWORD(v76) = v7;
  v9 = *(_QWORD *)v4;
  v69 = v8;
  v10 = *(_QWORD *)(*(_QWORD *)(v4 + 120) + 96LL);
  v11 = *(_QWORD *)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = *(_QWORD *)v11;
  v12 = 0;
  if ( v11 )
  {
    _BitScanReverse64(&v11, v11);
    v12 = 1;
    v64 = 63 - (v11 ^ 0x3F);
  }
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  *(_QWORD *)&v78 = v70 & 0xFFFFFFFFFFFFFFFBLL;
  v13 = 0;
  *((_QWORD *)&v78 + 1) = 0;
  BYTE4(v79) = 0;
  if ( v70 )
  {
    v14 = *(_QWORD *)(v70 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
    v13 = *(_DWORD *)(v14 + 8) >> 8;
  }
  LODWORD(v79) = v13;
  v71 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v66 + 32LL);
  v15 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v71 == sub_2D42F30 )
  {
    v16 = sub_AE2980(v15, 0)[1];
    v17 = 2;
    if ( v16 != 1 )
    {
      v17 = 3;
      if ( v16 != 2 )
      {
        v17 = 4;
        if ( v16 != 4 )
        {
          v17 = 5;
          if ( v16 != 8 )
          {
            v17 = 6;
            if ( v16 != 16 )
            {
              v17 = 7;
              if ( v16 != 32 )
              {
                v17 = 8;
                if ( v16 != 64 )
                  v17 = 9 * (v16 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v17 = v71((__int64)v66, v15, 0);
  }
  v19 = sub_33F1F00(
          (__int64 *)a1,
          v17,
          0,
          (__int64)&v74,
          v9,
          v8,
          v5.m128i_i64[0],
          v5.m128i_i64[1],
          v78,
          v79,
          0,
          0,
          (__int64)&v80,
          0);
  *(_QWORD *)&v72 = v19;
  v20 = (unsigned int)v18;
  *((_QWORD *)&v72 + 1) = v18;
  v21 = v19;
  if ( v12 && v64 > v66[73] )
  {
    v50 = 16LL * (unsigned int)v18;
    v68 = v19;
    *(_QWORD *)&v51 = sub_3400BD0(
                        a1,
                        (1LL << v64) - 1,
                        (__int64)&v74,
                        *(unsigned __int16 *)(v50 + v19[3].m128i_i64[0]),
                        *(_QWORD *)(v50 + v19[3].m128i_i64[0] + 8),
                        0,
                        v5,
                        0);
    *(_QWORD *)&v72 = sub_3406EB0(
                        (_QWORD *)a1,
                        0x38u,
                        (__int64)&v74,
                        *(unsigned __int16 *)(v68[3].m128i_i64[0] + v50),
                        *(_QWORD *)(v68[3].m128i_i64[0] + v50 + 8),
                        v52,
                        v72,
                        v51);
    v54 = 16LL * v53;
    *((_QWORD *)&v72 + 1) = v53 | *((_QWORD *)&v72 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v55 = sub_3401400(
                        a1,
                        -(1LL << v64),
                        (__int64)&v74,
                        *(unsigned __int16 *)(v54 + *(_QWORD *)(v72 + 48)),
                        *(_QWORD *)(v54 + *(_QWORD *)(v72 + 48) + 8),
                        0,
                        v5,
                        0);
    v57 = sub_3406EB0(
            (_QWORD *)a1,
            0xBAu,
            (__int64)&v74,
            *(unsigned __int16 *)(*(_QWORD *)(v72 + 48) + v54),
            *(_QWORD *)(*(_QWORD *)(v72 + 48) + v54 + 8),
            v56,
            v72,
            v55);
    v19 = v68;
    v21 = (__m128i *)v57;
    v20 = v58;
  }
  v22 = 16 * v20;
  v60 = (unsigned __int64)v19;
  v23 = (unsigned __int16 *)(16 * v20 + v21[3].m128i_i64[0]);
  v61 = (__int64)v21;
  v24 = *v23;
  v67 = v20;
  v62 = *((_QWORD *)v23 + 1);
  v25 = sub_2E79000(*(__int64 **)(a1 + 40));
  v63 = sub_3007410((__int64)&v76, *(__int64 **)(a1 + 64), v26, v27, v28, v29);
  v65 = sub_AE5020(v25, v63);
  v30 = sub_9208B0(v25, v63);
  v81 = v31;
  v80 = ((1LL << v65) + ((unsigned __int64)(v30 + 7) >> 3) - 1) >> v65 << v65;
  v32 = sub_CA1930(&v80);
  *(_QWORD *)&v33 = sub_3400BD0(a1, v32, (__int64)&v74, v24, v62, 0, v5, 0);
  v73 = v67 | *((_QWORD *)&v72 + 1) & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v59 + 1) = v73;
  *(_QWORD *)&v59 = v61;
  v37 = sub_3406EB0(
          (_QWORD *)a1,
          0x38u,
          (__int64)&v74,
          *(unsigned __int16 *)(*(_QWORD *)(v61 + 48) + v22),
          *(_QWORD *)(*(_QWORD *)(v61 + 48) + v22 + 8),
          v34,
          v59,
          v33);
  v38 = 0;
  v39 = v35;
  v80 = 0;
  v40 = v35 | v69 & 0xFFFFFFFF00000000LL;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  if ( v70 )
  {
    v41 = *(_QWORD *)(v70 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 <= 1 )
      v41 = **(_QWORD **)(v41 + 16);
    v38 = *(_DWORD *)(v41 + 8) >> 8;
  }
  LODWORD(v79) = v38;
  v42 = (unsigned __int16 *)(*((_QWORD *)v37 + 6) + 16LL * v35);
  v43 = *((_QWORD *)v42 + 1);
  v44 = *v42;
  *(_QWORD *)&v78 = v70 & 0xFFFFFFFFFFFFFFFBLL;
  *((_QWORD *)&v78 + 1) = 0;
  BYTE4(v79) = 0;
  v45 = sub_33CC4A0(a1, v44, v43, v70, v36, v39);
  v46 = sub_33F4560(
          (_QWORD *)a1,
          v60,
          1u,
          (__int64)&v74,
          (unsigned __int64)v37,
          v40,
          v5.m128i_u64[0],
          v5.m128i_u64[1],
          v78,
          v79,
          v45,
          0,
          (__int64)&v80);
  BYTE4(v79) = 0;
  LODWORD(v79) = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v78 = 0u;
  v48 = sub_33F1F00(
          (__int64 *)a1,
          v76,
          v77,
          (__int64)&v74,
          (__int64)v46,
          v40 & 0xFFFFFFFF00000000LL | v47,
          v61,
          v73,
          0,
          v79,
          0,
          0,
          (__int64)&v80,
          0);
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  return v48;
}
