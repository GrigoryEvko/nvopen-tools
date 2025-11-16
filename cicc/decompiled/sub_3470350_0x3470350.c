// Function: sub_3470350
// Address: 0x3470350
//
__int64 __fastcall sub_3470350(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        __int64 a4,
        __m128i *a5,
        __int64 *a6,
        __m128i a7,
        _DWORD *a8,
        __int128 a9,
        __int128 a10,
        bool *a11,
        __int64 a12,
        unsigned __int8 **a13,
        unsigned __int8 a14)
{
  unsigned __int16 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // r13
  int v18; // eax
  signed int v24; // eax
  int v25; // r9d
  __m128i v26; // xmm0
  int v27; // edx
  int v28; // edi
  signed int v29; // edi
  int v30; // eax
  int v31; // eax
  char v32; // r12
  int v33; // edx
  __m128i v34; // xmm0
  int v35; // r11d
  __int64 v36; // rsi
  __int64 v37; // r12
  __int64 v38; // r13
  __int128 v39; // rax
  __int64 v40; // r12
  __int64 v41; // r13
  unsigned int v42; // edx
  __int128 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // r9
  __int64 v46; // r12
  int v47; // edx
  __int32 v48; // edx
  __int64 v49; // r12
  __int64 v50; // r13
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned __int8 *v53; // rax
  __int64 v54; // r12
  __int64 v55; // r13
  unsigned int v56; // edx
  __int128 v57; // rax
  __int64 v58; // r9
  unsigned __int8 *v59; // rax
  unsigned int v60; // edx
  __int64 v61; // r9
  __int128 v62; // rax
  __int64 v63; // r9
  unsigned int v64; // edx
  unsigned __int8 *v65; // rsi
  __int32 v66; // edx
  __int128 v67; // rax
  __int64 v68; // r9
  __int128 v69; // rax
  __int64 v70; // r9
  __int128 v71; // rax
  __int64 v72; // r9
  unsigned __int8 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  unsigned int v76; // edx
  unsigned int v77; // edx
  __int128 v78; // [rsp-50h] [rbp-200h]
  __int128 v79; // [rsp-50h] [rbp-200h]
  __int128 v80; // [rsp-40h] [rbp-1F0h]
  __int128 v81; // [rsp-40h] [rbp-1F0h]
  __int128 v82; // [rsp-40h] [rbp-1F0h]
  __int128 v83; // [rsp-40h] [rbp-1F0h]
  __int128 v84; // [rsp-40h] [rbp-1F0h]
  __int128 v85; // [rsp-30h] [rbp-1E0h]
  __int128 v86; // [rsp-20h] [rbp-1D0h]
  __int128 v87; // [rsp-20h] [rbp-1D0h]
  __int128 v88; // [rsp-20h] [rbp-1D0h]
  __int128 v89; // [rsp-20h] [rbp-1D0h]
  __int128 v90; // [rsp-20h] [rbp-1D0h]
  __int128 v91; // [rsp-20h] [rbp-1D0h]
  __int128 v92; // [rsp-10h] [rbp-1C0h]
  __int128 v93; // [rsp-10h] [rbp-1C0h]
  __int128 v94; // [rsp-10h] [rbp-1C0h]
  __int128 v95; // [rsp-10h] [rbp-1C0h]
  __int128 v96; // [rsp-10h] [rbp-1C0h]
  __int128 v97; // [rsp-10h] [rbp-1C0h]
  __int128 v98; // [rsp+0h] [rbp-1B0h]
  unsigned int v99; // [rsp+0h] [rbp-1B0h]
  unsigned int v100; // [rsp+0h] [rbp-1B0h]
  char v101; // [rsp+10h] [rbp-1A0h]
  int v102; // [rsp+10h] [rbp-1A0h]
  __int64 v103; // [rsp+18h] [rbp-198h]
  int v104; // [rsp+20h] [rbp-190h]
  char v105; // [rsp+20h] [rbp-190h]
  __int64 v106; // [rsp+28h] [rbp-188h]
  __int128 v107; // [rsp+30h] [rbp-180h]
  __int64 v108; // [rsp+30h] [rbp-180h]
  unsigned int v109; // [rsp+40h] [rbp-170h]
  unsigned __int16 v110; // [rsp+44h] [rbp-16Ch]
  unsigned int v111; // [rsp+44h] [rbp-16Ch]

  v15 = *(_WORD *)(*(_QWORD *)(a5->m128i_i64[0] + 48) + 16LL * a5->m128i_u32[2]);
  v16 = *(int *)(*(_QWORD *)a8 + 96LL);
  v17 = v15 >> 3;
  *a11 = 0;
  v18 = (*(_DWORD *)(a1 + 4 * (35 * v16 + v17 + 130384)) >> (4 * (v15 & 7))) & 0xF;
  if ( v18 )
  {
    if ( (_BYTE)v18 != 2 )
      goto LABEL_55;
    v110 = v15;
    v101 = 4 * (v15 & 7);
    v104 = v16;
    v24 = sub_33CBD20(v16);
    v25 = v104;
    if ( ((*(_DWORD *)(a1 + 4 * (35LL * v24 + v17 + 130384)) >> v101) & 0xB) == 0 )
    {
      v26 = _mm_loadu_si128(a5);
      a5->m128i_i64[0] = *a6;
      a5->m128i_i32[2] = *((_DWORD *)a6 + 2);
      *a6 = v26.m128i_i64[0];
      *((_DWORD *)a6 + 2) = v26.m128i_i32[2];
      *(_QWORD *)a8 = sub_33ED040(a2, v24);
      a8[2] = v27;
      return 1;
    }
    v28 = v104;
    v105 = v101;
    v102 = v25;
    v29 = sub_33CBD40(v28, v110, 0);
    v30 = (*(_DWORD *)(a1 + 4 * (35LL * v29 + v17 + 130384)) >> v105) & 0xF;
    if ( !v30 || (_BYTE)v30 == 4 )
    {
      v32 = 0;
    }
    else
    {
      v29 = sub_33CBD20(v29);
      v31 = (*(_DWORD *)(a1 + 4 * (35LL * v29 + v17 + 130384)) >> v105) & 0xF;
      if ( v31 && (_BYTE)v31 != 4 )
      {
        if ( v110 == 2 )
        {
          v61 = (unsigned int)(v102 - 10);
          switch ( v102 )
          {
            case 10:
            case 20:
              *(_QWORD *)&v71 = sub_34074A0(a2, a12, *a6, a6[1], 2u, 0, a7);
              v65 = sub_3406EB0(a2, 0xBAu, a12, 2, 0, v72, (__int128)*a5, v71);
              goto LABEL_44;
            case 11:
            case 21:
              *(_QWORD *)&v69 = sub_34074A0(a2, a12, *a6, a6[1], 2u, 0, a7);
              v65 = sub_3406EB0(a2, 0xBBu, a12, 2, 0, v70, (__int128)*a5, v69);
              goto LABEL_44;
            case 12:
            case 18:
              *(_QWORD *)&v67 = sub_34074A0(a2, a12, a5->m128i_i64[0], a5->m128i_i64[1], 2u, 0, a7);
              v65 = sub_3406EB0(a2, 0xBAu, a12, 2, 0, v68, *(_OWORD *)a6, v67);
              goto LABEL_44;
            case 13:
            case 19:
              *(_QWORD *)&v62 = sub_34074A0(a2, a12, a5->m128i_i64[0], a5->m128i_i64[1], 2u, 0, a7);
              v65 = sub_3406EB0(a2, 0xBBu, a12, 2, 0, v63, *(_OWORD *)a6, v62);
              goto LABEL_44;
            case 17:
              v73 = sub_3406EB0(a2, 0xBCu, a12, 2, 0, v61, (__int128)*a5, *(_OWORD *)a6);
              v65 = sub_34074A0(a2, a12, (__int64)v73, v74, 2u, 0, a7);
              goto LABEL_44;
            case 22:
              v65 = sub_3406EB0(a2, 0xBCu, a12, 2, 0, v61, (__int128)*a5, *(_OWORD *)a6);
LABEL_44:
              a5->m128i_i64[0] = (__int64)sub_33FB310((__int64)a2, (__int64)v65, v64, a12, a3, a4, a7);
              a5->m128i_i32[2] = v66;
              goto LABEL_30;
            default:
              goto LABEL_55;
          }
        }
        switch ( v102 )
        {
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
          case 10:
          case 11:
          case 12:
          case 13:
          case 14:
            goto LABEL_17;
          case 6:
          case 9:
            if ( ((*(_DWORD *)(a1 + 4 * (35LL * (((v102 & 8) != 0) + 7) + v17 + 130384)) >> v105) & 0xF) != 0
              && (((*(_DWORD *)(a1 + 4 * v17 + 521816) >> v105) & 0xF) == 0
               || ((*(_DWORD *)(a1 + 4 * v17 + 522096) >> v105) & 0xF) == 0) )
            {
              v35 = 4;
              v111 = 187;
              v36 = 2;
              *a11 = (v102 & 8) != 0;
LABEL_21:
              if ( (unsigned int)(v102 - 7) <= 1 )
              {
LABEL_38:
                v99 = v35;
                if ( (_QWORD)a10 )
                {
                  v49 = a5->m128i_i64[0];
                  v50 = a5->m128i_i64[1];
                  *(_QWORD *)&v51 = sub_33ED040(a2, v36);
                  *((_QWORD *)&v83 + 1) = v50;
                  *(_QWORD *)&v83 = v49;
                  *((_QWORD *)&v78 + 1) = v50;
                  *(_QWORD *)&v78 = v49;
                  v53 = sub_33FC1D0(a2, 463, a12, a3, a4, v52, v78, v83, v51, a9, a10);
                  v54 = *a6;
                  v55 = a6[1];
                  v108 = (__int64)v53;
                  v106 = v56;
                  *(_QWORD *)&v57 = sub_33ED040(a2, v99);
                  *((_QWORD *)&v84 + 1) = v55;
                  *(_QWORD *)&v84 = v54;
                  *((_QWORD *)&v79 + 1) = v55;
                  *(_QWORD *)&v79 = v54;
                  v59 = sub_33FC1D0(a2, 463, a12, a3, a4, v58, v79, v84, v57, a9, a10);
                  v45 = 0xFFFFFFFF00000000LL;
                  v46 = (__int64)v59;
                }
                else
                {
                  *((_QWORD *)&v96 + 1) = a14;
                  *(_QWORD *)&v96 = a13[1];
                  *((_QWORD *)&v90 + 1) = *a13;
                  *(_QWORD *)&v90 = v36;
                  v108 = sub_32889F0(
                           (__int64)a2,
                           a12,
                           a3,
                           a4,
                           a5->m128i_i64[0],
                           a5->m128i_i64[1],
                           (__int128)*a5,
                           v90,
                           v96);
                  *((_QWORD *)&v97 + 1) = a14;
                  v106 = v77;
                  *(_QWORD *)&v97 = a13[1];
                  *((_QWORD *)&v91 + 1) = *a13;
                  *(_QWORD *)&v91 = v99;
                  v46 = sub_32889F0((__int64)a2, a12, a3, a4, *a6, a6[1], *(_OWORD *)a6, v91, v97);
                }
                v103 = v60;
              }
              else
              {
                if ( (_QWORD)a10 )
                {
                  v37 = *a6;
                  v109 = v35;
                  v38 = a6[1];
                  v107 = (__int128)*a5;
                  *(_QWORD *)&v39 = sub_33ED040(a2, v36);
                  *((_QWORD *)&v80 + 1) = v38;
                  *(_QWORD *)&v80 = v37;
                  v108 = (__int64)sub_33FC1D0(a2, 463, a12, a3, a4, *((__int64 *)&v107 + 1), v107, v80, v39, a9, a10);
                  v40 = *a6;
                  v41 = a6[1];
                  v98 = (__int128)*a5;
                  v106 = v42;
                  *(_QWORD *)&v43 = sub_33ED040(a2, v109);
                  *((_QWORD *)&v81 + 1) = v41;
                  *(_QWORD *)&v81 = v40;
                  v46 = (__int64)sub_33FC1D0(a2, 463, a12, a3, a4, *((__int64 *)&v98 + 1), v98, v81, v43, a9, a10);
                }
                else
                {
                  v100 = v35;
                  *((_QWORD *)&v93 + 1) = a14;
                  *(_QWORD *)&v93 = a13[1];
                  *((_QWORD *)&v87 + 1) = *a13;
                  *(_QWORD *)&v87 = v36;
                  v75 = sub_32889F0(
                          (__int64)a2,
                          a12,
                          a3,
                          a4,
                          a5->m128i_i64[0],
                          a5->m128i_i64[1],
                          *(_OWORD *)a6,
                          v87,
                          v93);
                  *((_QWORD *)&v94 + 1) = a14;
                  v108 = v75;
                  v106 = v76;
                  *(_QWORD *)&v94 = a13[1];
                  *((_QWORD *)&v88 + 1) = *a13;
                  *(_QWORD *)&v88 = v100;
                  v46 = sub_32889F0(
                          (__int64)a2,
                          a12,
                          a3,
                          a4,
                          a5->m128i_i64[0],
                          a5->m128i_i64[1],
                          *(_OWORD *)a6,
                          v88,
                          v94);
                }
                v103 = v44;
              }
              if ( *a13 )
              {
                *((_QWORD *)&v92 + 1) = 1;
                *(_QWORD *)&v92 = v46;
                *((_QWORD *)&v86 + 1) = 1;
                *(_QWORD *)&v86 = v108;
                *a13 = sub_3406EB0(a2, 2u, a12, 1, 0, v45, v86, v92);
                *((_DWORD *)a13 + 2) = v47;
              }
              if ( (_QWORD)a10 )
              {
                *((_QWORD *)&v85 + 1) = v103;
                *(_QWORD *)&v85 = v46;
                *((_QWORD *)&v82 + 1) = v106;
                *(_QWORD *)&v82 = v108;
                a5->m128i_i64[0] = (__int64)sub_33FC130(
                                              a2,
                                              4 * (unsigned int)(v111 == 187) + 396,
                                              a12,
                                              a3,
                                              a4,
                                              v45,
                                              v82,
                                              v85,
                                              a9,
                                              a10);
              }
              else
              {
                *((_QWORD *)&v95 + 1) = v103;
                *(_QWORD *)&v95 = v46;
                *((_QWORD *)&v89 + 1) = v106;
                *(_QWORD *)&v89 = v108;
                a5->m128i_i64[0] = (__int64)sub_3406EB0(a2, v111, a12, a3, a4, v45, v89, v95);
              }
              a5->m128i_i32[2] = v48;
LABEL_30:
              *a6 = 0;
              *((_DWORD *)a6 + 2) = 0;
              *(_QWORD *)a8 = 0;
              a8[2] = 0;
              return 1;
            }
LABEL_17:
            if ( (unsigned __int16)(v110 - 2) > 7u
              && (unsigned __int16)(v110 - 17) > 0x6Cu
              && (unsigned __int16)(v110 - 176) > 0x1Fu )
            {
              v35 = 8 - ((v102 & 8) == 0);
              v36 = v102 & 7 | 0x10u;
              v111 = 187 - ((v102 & 8) == 0);
              goto LABEL_21;
            }
            break;
          case 7:
            v111 = 186;
            v35 = 1;
            v36 = 1;
            goto LABEL_21;
          case 8:
            if ( ((*(_DWORD *)(a1 + 4 * v17 + 523496) >> v105) & 0xF) != 0 )
            {
              v36 = 1;
              v111 = 186;
              *a11 = 1;
            }
            else
            {
              v111 = 187;
              v36 = 14;
            }
            v35 = v36;
            goto LABEL_38;
          default:
            break;
        }
LABEL_55:
        BUG();
      }
      v32 = 1;
    }
    *(_QWORD *)a8 = sub_33ED040(a2, v29);
    a8[2] = v33;
    *a11 = 1;
    if ( v32 )
    {
      v34 = _mm_loadu_si128(a5);
      a5->m128i_i64[0] = *a6;
      a5->m128i_i32[2] = *((_DWORD *)a6 + 2);
      *a6 = v34.m128i_i64[0];
      *((_DWORD *)a6 + 2) = v34.m128i_i32[2];
    }
    return 1;
  }
  return 0;
}
