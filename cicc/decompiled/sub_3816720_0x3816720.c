// Function: sub_3816720
// Address: 0x3816720
//
__int64 __fastcall sub_3816720(__int64 **a1, unsigned __int64 a2, __m128i a3)
{
  int v3; // r13d
  __int64 v5; // rax
  __int128 v6; // xmm1
  __int64 v7; // rdi
  __int16 *v8; // rax
  __int64 v9; // rsi
  unsigned __int16 v10; // r15
  unsigned __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 (__fastcall *v14)(__int64, __int64, _QWORD, unsigned __int64, const __m128i *); // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rdx
  const __m128i *v18; // rax
  const __m128i *i; // rdx
  unsigned int v20; // esi
  __int64 v21; // r8
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // r11
  _QWORD *v26; // rax
  unsigned __int8 *v27; // rax
  int v28; // edx
  int v29; // edi
  unsigned __int8 *v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // r9
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v34; // rax
  unsigned __int16 v35; // si
  __int64 v36; // r8
  __int64 *v37; // rax
  int v38; // r9d
  __int64 v39; // r15
  int v40; // r9d
  __int64 v41; // rdx
  __int64 v42; // r12
  __int128 v43; // rax
  __int64 v44; // r13
  __int64 v45; // rdi
  int v46; // edx
  unsigned __int16 v47; // ax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int128 v50; // rax
  __int64 v51; // r9
  unsigned int v52; // edx
  __int64 v53; // r9
  unsigned int v54; // edx
  __int64 (__fastcall *v56)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 (__fastcall *v59)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rdx
  const __m128i *v62; // rdx
  __m128i *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __int128 v68; // [rsp-10h] [rbp-1D0h]
  __int64 v69; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v70; // [rsp+10h] [rbp-1B0h]
  __int64 v71; // [rsp+20h] [rbp-1A0h]
  __int64 v73; // [rsp+30h] [rbp-190h]
  unsigned int v74; // [rsp+38h] [rbp-188h]
  __int64 (__fastcall *v75)(__int64, __int64, unsigned int); // [rsp+38h] [rbp-188h]
  __int64 v76; // [rsp+38h] [rbp-188h]
  unsigned __int64 v77; // [rsp+38h] [rbp-188h]
  unsigned __int16 v78; // [rsp+40h] [rbp-180h]
  unsigned int v79; // [rsp+48h] [rbp-178h]
  unsigned __int64 v80; // [rsp+50h] [rbp-170h]
  __int128 v81; // [rsp+50h] [rbp-170h]
  __int128 v82; // [rsp+60h] [rbp-160h]
  unsigned __int8 *v83; // [rsp+80h] [rbp-140h]
  unsigned __int16 v84; // [rsp+BAh] [rbp-106h] BYREF
  unsigned int v85; // [rsp+BCh] [rbp-104h] BYREF
  __int64 v86; // [rsp+C0h] [rbp-100h] BYREF
  int v87; // [rsp+C8h] [rbp-F8h]
  __int64 v88; // [rsp+D0h] [rbp-F0h] BYREF
  unsigned __int64 v89; // [rsp+D8h] [rbp-E8h]
  __int64 v90; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v91; // [rsp+E8h] [rbp-D8h]
  __int64 v92; // [rsp+F0h] [rbp-D0h]
  const __m128i *v93; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v94; // [rsp+108h] [rbp-B8h]
  _QWORD v95[22]; // [rsp+110h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 + 40);
  v6 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 40));
  v70 = *(_QWORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *(_QWORD *)(a2 + 80);
  v69 = v7;
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v86 = v9;
  v80 = v11;
  if ( v9 )
    sub_B96E90((__int64)&v86, v9, 1);
  v12 = (__int64)*a1;
  LOWORD(v88) = v10;
  v87 = *(_DWORD *)(a2 + 72);
  v13 = a1[1][8];
  v89 = v80;
  if ( v10 )
  {
    v78 = *(_WORD *)(v12 + 2LL * v10 + 2852);
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v88) )
  {
    if ( !sub_3007070((__int64)&v88) )
      goto LABEL_64;
    v56 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v12 + 592LL);
    if ( v56 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v93, v12, v13, v88, v89);
      v57 = v95[0];
      v58 = (unsigned __int16)v94;
    }
    else
    {
      v58 = v56(v12, v13, v88, v80);
      v57 = v65;
    }
    v90 = v58;
    v91 = v57;
    if ( (_WORD)v58 )
    {
      v78 = *(_WORD *)(v12 + 2LL * (unsigned __int16)v58 + 2852);
    }
    else
    {
      v77 = v57;
      if ( !sub_30070B0((__int64)&v90) )
      {
        if ( sub_3007070((__int64)&v90) )
        {
          v59 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v12 + 592LL);
          if ( v59 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v93, v12, v13, v90, v91);
            v60 = v95[0];
            v61 = (unsigned __int16)v94;
          }
          else
          {
            v66 = v59(v12, v13, v90, v77);
            v60 = v67;
            v61 = v66;
          }
          v78 = sub_2FE98B0(v12, v13, v61, v60);
          goto LABEL_50;
        }
LABEL_64:
        BUG();
      }
      v84 = 0;
      LOWORD(v93) = 0;
      v94 = 0;
      sub_2FE8D10(v12, v13, (unsigned int)v90, v77, (__int64 *)&v93, &v85, &v84);
      v78 = v84;
    }
LABEL_50:
    v12 = (__int64)*a1;
    v13 = a1[1][8];
    goto LABEL_5;
  }
  LOWORD(v85) = 0;
  LOWORD(v93) = 0;
  v94 = 0;
  sub_2FE8D10(v12, v13, (unsigned int)v88, v80, (__int64 *)&v93, (unsigned int *)&v90, (unsigned __int16 *)&v85);
  v12 = (__int64)*a1;
  v78 = v85;
  v13 = a1[1][8];
LABEL_5:
  v14 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, unsigned __int64, const __m128i *))(*(_QWORD *)v12 + 736LL);
  BYTE2(v93) = 0;
  v74 = v14(v12, v13, v10, v80, v93);
  v93 = (const __m128i *)v95;
  v71 = v74;
  v94 = 0x800000000LL;
  if ( v74 )
  {
    v17 = v95;
    v18 = (const __m128i *)v95;
    if ( v74 > 8uLL )
    {
      sub_C8D5F0((__int64)&v93, v95, v74, 0x10u, v15, v16);
      v17 = (__int64 *)v93;
      v18 = &v93[(unsigned int)v94];
    }
    for ( i = (const __m128i *)&v17[2 * v74]; i != v18; ++v18 )
    {
      if ( v18 )
      {
        v18->m128i_i64[0] = 0;
        v18->m128i_i32[2] = 0;
      }
    }
    HIWORD(v20) = HIWORD(v3);
    v21 = v70;
    v22 = 0;
    v23 = v7;
    LODWORD(v94) = v74;
    do
    {
      v24 = *(_QWORD *)(a2 + 40);
      v25 = *(_QWORD *)(*(_QWORD *)(v24 + 120) + 96LL);
      v26 = *(_QWORD **)(v25 + 24);
      if ( *(_DWORD *)(v25 + 32) > 0x40u )
        v26 = (_QWORD *)*v26;
      LOWORD(v20) = v78;
      v27 = sub_3411830(a1[1], v20, 0, (__int64)&v86, v21, v23, v6, *(_OWORD *)(v24 + 80), (unsigned int)v26);
      v29 = v28;
      v30 = v27;
      v31 = (__int64)v93;
      v32 = v23 & 0xFFFFFFFF00000000LL | 1;
      v23 = v32;
      v93[v22].m128i_i64[0] = (__int64)v30;
      *(_DWORD *)(v31 + v22 * 16 + 8) = v29;
      v21 = v93[v22++].m128i_i64[0];
    }
    while ( v74 != v22 );
    v70 = v21;
    v69 = v32;
  }
  if ( *(_BYTE *)sub_2E79000((__int64 *)a1[1][5]) )
  {
    v62 = v93;
    v63 = (__m128i *)&v93[(unsigned int)v94];
    if ( v93 != v63 )
    {
      while ( v62 < --v63 )
      {
        a3 = _mm_loadu_si128(v62++);
        v62[-1].m128i_i64[0] = v63->m128i_i64[0];
        v62[-1].m128i_i32[2] = v63->m128i_i32[2];
        v63->m128i_i64[0] = a3.m128i_i64[0];
        v63->m128i_i32[2] = a3.m128i_i32[2];
      }
    }
  }
  v33 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**a1 + 592);
  v34 = *(__int16 **)(a2 + 48);
  v35 = *v34;
  v36 = *((_QWORD *)v34 + 1);
  v37 = a1[1];
  if ( v33 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v90, (__int64)*a1, v37[8], v35, v36);
    v79 = (unsigned __int16)v91;
    v39 = v92;
  }
  else
  {
    v79 = v33((__int64)*a1, v37[8], v35, v36);
    v39 = v64;
  }
  *(_QWORD *)&v81 = sub_33FAF80((__int64)a1[1], 214, (__int64)&v86, v79, v39, v38, a3);
  *((_QWORD *)&v81 + 1) = v41;
  if ( v74 > 1 )
  {
    v42 = 1;
    do
    {
      *(_QWORD *)&v43 = sub_33FAF80((__int64)a1[1], 214, (__int64)&v86, v79, v39, v40, a3);
      v44 = (__int64)a1[1];
      v82 = v43;
      v73 = (__int64)*a1;
      v75 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(**a1 + 32);
      v45 = sub_2E79000(*(__int64 **)(v44 + 40));
      if ( v75 == sub_2D42F30 )
      {
        v46 = sub_AE2980(v45, 0)[1];
        v47 = 2;
        if ( v46 != 1 )
        {
          v47 = 3;
          if ( v46 != 2 )
          {
            v47 = 4;
            if ( v46 != 4 )
            {
              v47 = 5;
              if ( v46 != 8 )
              {
                v47 = 6;
                if ( v46 != 16 )
                {
                  v47 = 7;
                  if ( v46 != 32 )
                  {
                    v47 = 8;
                    if ( v46 != 64 )
                      v47 = 9 * (v46 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v47 = v75(v73, v45, 0);
      }
      if ( v78 <= 1u || (unsigned __int16)(v78 - 504) <= 7u )
        BUG();
      v76 = v47;
      v48 = v42 * *(_QWORD *)&byte_444C4A0[16 * v78 - 16];
      LOBYTE(v91) = byte_444C4A0[16 * v78 - 8];
      ++v42;
      v90 = v48;
      v49 = sub_CA1930(&v90);
      *(_QWORD *)&v50 = sub_3400BD0(v44, v49, (__int64)&v86, v76, 0, 0, a3, 0);
      v83 = sub_3406EB0((_QWORD *)v44, 0xBEu, (__int64)&v86, v79, v39, v51, v82, v50);
      *((_QWORD *)&v68 + 1) = v52 | *((_QWORD *)&v82 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v68 = v83;
      *(_QWORD *)&v81 = sub_3406EB0(a1[1], 0xBBu, (__int64)&v86, v79, v39, v53, v81, v68);
      *((_QWORD *)&v81 + 1) = v54 | *((_QWORD *)&v81 + 1) & 0xFFFFFFFF00000000LL;
    }
    while ( v42 != v71 );
  }
  sub_3760E70((__int64)a1, a2, 1, v70, v69);
  if ( v93 != (const __m128i *)v95 )
    _libc_free((unsigned __int64)v93);
  if ( v86 )
    sub_B91220((__int64)&v86, v86);
  return v81;
}
