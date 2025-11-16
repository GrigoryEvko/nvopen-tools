// Function: sub_3494EA0
// Address: 0x3494ea0
//
__m128i *__fastcall sub_3494EA0(
        _WORD *a1,
        __int64 a2,
        __int16 a3,
        __int64 a4,
        __m128i *a5,
        __m128i *a6,
        unsigned int *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 *a13)
{
  unsigned int v14; // edi
  _BOOL4 v15; // r14d
  _BOOL4 v16; // ecx
  int v17; // r14d
  int v18; // ecx
  _BOOL4 v19; // r14d
  _BOOL4 v20; // ecx
  _BOOL4 v21; // r14d
  _BOOL4 v22; // ecx
  _BOOL4 v23; // r14d
  _BOOL4 v24; // ecx
  _BOOL4 v25; // r14d
  _BOOL4 v26; // ecx
  _BOOL4 v27; // r14d
  _BOOL4 v28; // ecx
  _BOOL4 v29; // r14d
  _BOOL4 v30; // ecx
  _BOOL4 v31; // r14d
  _BOOL4 v32; // ecx
  _BOOL4 v33; // r14d
  _BOOL4 v34; // ecx
  _BOOL4 v35; // r14d
  _BOOL4 v36; // ecx
  int v37; // eax
  __int16 v38; // ax
  __int64 v39; // r13
  __m128i v40; // xmm0
  __m128i v41; // xmm2
  __int64 v42; // r15
  __int16 v43; // di
  __int64 v44; // rax
  unsigned int v45; // r13d
  __int16 v46; // dx
  __int32 v47; // edx
  int v48; // edi
  __m128i *result; // rax
  __int64 v50; // r15
  __int64 (__fastcall *v51)(_WORD *, __int64, __int64, _QWORD, _QWORD); // r14
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r14
  __int64 v55; // r15
  __int128 v56; // rax
  __int128 v57; // rax
  unsigned int v58; // r8d
  __int64 v59; // r14
  __int64 v60; // r15
  __int128 v61; // rax
  __int64 v62; // r9
  __int64 v63; // r9
  __int32 v64; // edx
  int v65; // edx
  __int32 v66; // edx
  _BOOL4 v67; // r14d
  _BOOL4 v68; // ecx
  _BOOL4 v69; // r14d
  _BOOL4 v70; // ecx
  __int128 v71; // [rsp-30h] [rbp-1B0h]
  __int128 v72; // [rsp-30h] [rbp-1B0h]
  __int64 v73; // [rsp-10h] [rbp-190h]
  __int64 v74; // [rsp-8h] [rbp-188h]
  unsigned int v75; // [rsp+0h] [rbp-180h]
  __int64 v76; // [rsp+8h] [rbp-178h]
  unsigned int v77; // [rsp+10h] [rbp-170h]
  int v78; // [rsp+18h] [rbp-168h]
  __int64 v79; // [rsp+18h] [rbp-168h]
  __int128 v80; // [rsp+20h] [rbp-160h]
  __int128 v81; // [rsp+20h] [rbp-160h]
  int v82; // [rsp+30h] [rbp-150h]
  __int128 v83; // [rsp+30h] [rbp-150h]
  char v84; // [rsp+4Fh] [rbp-131h]
  _OWORD v87[2]; // [rsp+A0h] [rbp-E0h] BYREF
  __int16 v88; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+C8h] [rbp-B8h]
  __int16 v90; // [rsp+D0h] [rbp-B0h]
  __int64 v91; // [rsp+D8h] [rbp-A8h]
  __int64 v92; // [rsp+E0h] [rbp-A0h] BYREF
  __int32 v93; // [rsp+E8h] [rbp-98h]
  __int128 v94; // [rsp+F0h] [rbp-90h]
  _QWORD v95[2]; // [rsp+100h] [rbp-80h] BYREF
  __int128 v96; // [rsp+110h] [rbp-70h]
  __int16 *v97; // [rsp+120h] [rbp-60h]
  __int64 v98; // [rsp+128h] [rbp-58h]
  __int64 v99; // [rsp+130h] [rbp-50h]
  __int64 v100; // [rsp+138h] [rbp-48h]
  __int64 v101; // [rsp+140h] [rbp-40h]

  v14 = *a7;
  switch ( *a7 )
  {
    case 1u:
    case 0x11u:
      if ( a3 == 12 )
      {
        v84 = 0;
        v17 = 440;
        v18 = 440;
        v82 = 729;
      }
      else
      {
        v84 = 0;
        if ( a3 == 13 )
        {
          v17 = 441;
          v18 = 441;
          v82 = 729;
        }
        else
        {
          v82 = 729;
          v19 = a3 != 15;
          v20 = v19;
          v17 = v19 + 442;
          v18 = v20 + 442;
        }
      }
      goto LABEL_39;
    case 2u:
    case 0x12u:
      if ( a3 == 12 )
      {
        v84 = 0;
        v17 = 460;
        v18 = 460;
        v82 = 729;
      }
      else
      {
        v84 = 0;
        if ( a3 == 13 )
        {
          v17 = 461;
          v18 = 461;
          v82 = 729;
        }
        else
        {
          v82 = 729;
          v21 = a3 != 15;
          v22 = v21;
          v17 = v21 + 462;
          v18 = v22 + 462;
        }
      }
      goto LABEL_39;
    case 3u:
    case 0x13u:
      if ( a3 == 12 )
      {
        v84 = 0;
        v17 = 448;
        v18 = 448;
        v82 = 729;
      }
      else
      {
        v84 = 0;
        if ( a3 == 13 )
        {
          v17 = 449;
          v18 = 449;
          v82 = 729;
        }
        else
        {
          v82 = 729;
          v23 = a3 != 15;
          v24 = v23;
          v17 = v23 + 450;
          v18 = v24 + 450;
        }
      }
      goto LABEL_39;
    case 4u:
    case 0x14u:
      if ( a3 == 12 )
      {
        v84 = 0;
        v17 = 452;
        v18 = 452;
        v82 = 729;
      }
      else
      {
        v84 = 0;
        if ( a3 == 13 )
        {
          v17 = 453;
          v18 = 453;
          v82 = 729;
        }
        else
        {
          v82 = 729;
          v25 = a3 != 15;
          v26 = v25;
          v17 = v25 + 454;
          v18 = v26 + 454;
        }
      }
      goto LABEL_39;
    case 5u:
    case 0x15u:
      if ( a3 == 12 )
      {
        v84 = 0;
        v17 = 456;
        v18 = 456;
        v82 = 729;
      }
      else
      {
        v84 = 0;
        if ( a3 == 13 )
        {
          v17 = 457;
          v18 = 457;
          v82 = 729;
        }
        else
        {
          v82 = 729;
          v27 = a3 != 15;
          v28 = v27;
          v17 = v27 + 458;
          v18 = v28 + 458;
        }
      }
      goto LABEL_39;
    case 6u:
      v84 = 1;
      goto LABEL_35;
    case 7u:
      v84 = 1;
      goto LABEL_31;
    case 8u:
      v84 = 0;
LABEL_31:
      if ( a3 == 12 )
      {
        v82 = 729;
        v17 = 464;
        v18 = 464;
      }
      else
      {
        v82 = 729;
        if ( a3 == 13 )
        {
          v17 = 465;
          v18 = 465;
        }
        else
        {
          v33 = a3 != 15;
          v34 = v33;
          v17 = v33 + 466;
          v18 = v34 + 466;
        }
      }
      goto LABEL_39;
    case 9u:
      v84 = 0;
LABEL_35:
      if ( a3 == 12 )
      {
        v82 = 440;
        v17 = 464;
        v18 = 464;
      }
      else if ( a3 == 13 )
      {
        v82 = 441;
        v17 = 465;
        v18 = 465;
      }
      else
      {
        v35 = a3 != 15;
        v36 = v35;
        v37 = v35 + 442;
        v17 = v35 + 466;
        v18 = v36 + 466;
        v82 = v37;
      }
      goto LABEL_39;
    case 0xEu:
    case 0x16u:
      if ( a3 == 12 )
      {
        v84 = 0;
        v17 = 444;
        v18 = 444;
        v82 = 729;
      }
      else
      {
        v84 = 0;
        if ( a3 == 13 )
        {
          v17 = 445;
          v18 = 445;
          v82 = 729;
        }
        else
        {
          v82 = 729;
          v29 = a3 != 15;
          v30 = v29;
          v17 = v29 + 446;
          v18 = v30 + 446;
        }
      }
      goto LABEL_39;
    default:
      if ( v14 == 12 )
      {
        if ( a3 == 12 )
        {
          v84 = 1;
          v17 = 448;
          v18 = 448;
          v82 = 729;
        }
        else
        {
          v84 = 1;
          if ( a3 == 13 )
          {
            v17 = 449;
            v18 = 449;
            v82 = 729;
          }
          else
          {
            v82 = 729;
            v69 = a3 != 15;
            v70 = v69;
            v17 = v69 + 450;
            v18 = v70 + 450;
          }
        }
      }
      else
      {
        if ( v14 <= 0xC )
        {
          if ( v14 == 10 )
          {
            if ( a3 == 12 )
            {
              v84 = 1;
              v17 = 456;
              v18 = 456;
              v82 = 729;
            }
            else
            {
              v84 = 1;
              if ( a3 == 13 )
              {
                v17 = 457;
                v18 = 457;
                v82 = 729;
              }
              else
              {
                v82 = 729;
                v15 = a3 != 15;
                v16 = v15;
                v17 = v15 + 458;
                v18 = v16 + 458;
              }
            }
            goto LABEL_39;
          }
          if ( v14 == 11 )
          {
            if ( a3 == 12 )
            {
              v84 = 1;
              v17 = 452;
              v18 = 452;
              v82 = 729;
            }
            else
            {
              v84 = 1;
              if ( a3 == 13 )
              {
                v17 = 453;
                v18 = 453;
                v82 = 729;
              }
              else
              {
                v82 = 729;
                v31 = a3 != 15;
                v32 = v31;
                v17 = v31 + 454;
                v18 = v32 + 454;
              }
            }
            goto LABEL_39;
          }
LABEL_80:
          BUG();
        }
        if ( v14 != 13 )
          goto LABEL_80;
        if ( a3 == 12 )
        {
          v84 = 1;
          v17 = 460;
          v18 = 460;
          v82 = 729;
        }
        else
        {
          v84 = 1;
          if ( a3 == 13 )
          {
            v17 = 461;
            v18 = 461;
            v82 = 729;
          }
          else
          {
            v82 = 729;
            v67 = a3 != 15;
            v68 = v67;
            v17 = v67 + 462;
            v18 = v68 + 462;
          }
        }
      }
LABEL_39:
      v78 = v18;
      v38 = (*(__int64 (__fastcall **)(_WORD *))(*(_QWORD *)a1 + 536LL))(a1);
      v39 = *(_QWORD *)(a9 + 48) + 16LL * (unsigned int)a10;
      LOBYTE(v101) = 20;
      v40 = _mm_loadu_si128(a5);
      LOWORD(v99) = v38;
      v41 = _mm_loadu_si128(a6);
      v97 = &v88;
      v87[0] = v40;
      v87[1] = v41;
      v42 = *(_QWORD *)(a11 + 48) + 16LL * (unsigned int)a12;
      v43 = *(_WORD *)v39;
      v89 = *(_QWORD *)(v39 + 8);
      v44 = *(_QWORD *)(v42 + 8);
      v45 = (unsigned __int16)v99;
      v88 = v43;
      v46 = *(_WORD *)v42;
      v91 = v44;
      v90 = v46;
      v74 = a13[1];
      v73 = *a13;
      v100 = 0;
      v98 = 2;
      sub_3494590(
        (__int64)&v92,
        a1,
        a2,
        v78,
        (unsigned __int16)v99,
        0,
        (__int64)v87,
        2u,
        (__int64)&v88,
        2,
        v99,
        0,
        20,
        a8,
        v73,
        v74);
      a5->m128i_i64[0] = v92;
      a5->m128i_i32[2] = v93;
      a6->m128i_i64[0] = (__int64)sub_3400BD0(a2, 0, a8, v45, 0, 0, v40, 0);
      a6->m128i_i32[2] = v47;
      v48 = *(_DWORD *)&a1[2 * v17 + 267024];
      *a7 = v48;
      if ( v84 )
        *a7 = sub_33CBD40(v48, v45, 0);
      if ( v82 == 729 )
      {
        *a13 = v94;
        result = (__m128i *)DWORD2(v94);
        *((_DWORD *)a13 + 2) = DWORD2(v94);
      }
      else
      {
        v50 = *(_QWORD *)(a2 + 64);
        v51 = *(__int64 (__fastcall **)(_WORD *, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 528LL);
        v52 = sub_2E79000(*(__int64 **)(a2 + 40));
        v77 = v51(a1, v52, v50, v45, 0);
        v79 = v53;
        v54 = a5->m128i_i64[0];
        v55 = a5->m128i_i64[1];
        v80 = (__int128)*a6;
        *(_QWORD *)&v56 = sub_33ED040((_QWORD *)a2, *a7);
        *((_QWORD *)&v71 + 1) = v55;
        *(_QWORD *)&v71 = v54;
        *(_QWORD *)&v57 = sub_340F900((_QWORD *)a2, 0xD0u, a8, v77, v79, *((__int64 *)&v80 + 1), v71, v80, v56);
        v76 = v57;
        v81 = v57;
        v75 = DWORD2(v57);
        sub_3494590(
          (__int64)v95,
          a1,
          a2,
          v82,
          v45,
          0,
          (__int64)v87,
          2u,
          (__int64)v97,
          v98,
          v99,
          v100,
          v101,
          a8,
          *a13,
          a13[1]);
        v58 = *(_DWORD *)&a1[2 * v82 + 267024];
        *a7 = v58;
        if ( v84 )
        {
          v58 = sub_33CBD40(v58, v45, 0);
          *a7 = v58;
        }
        v59 = v95[0];
        v60 = v95[1];
        v83 = (__int128)*a6;
        *(_QWORD *)&v61 = sub_33ED040((_QWORD *)a2, v58);
        *((_QWORD *)&v72 + 1) = v60;
        *(_QWORD *)&v72 = v59;
        a5->m128i_i64[0] = sub_340F900((_QWORD *)a2, 0xD0u, a8, v77, v79, v62, v72, v83, v61);
        a5->m128i_i32[2] = v64;
        if ( *a13 )
        {
          *a13 = (__int64)sub_3406EB0((_QWORD *)a2, 2u, a8, 1, 0, v63, v94, v96);
          *((_DWORD *)a13 + 2) = v65;
        }
        a5->m128i_i64[0] = (__int64)sub_3406EB0(
                                      (_QWORD *)a2,
                                      (unsigned int)(v84 == 0) + 186,
                                      a8,
                                      *(unsigned __int16 *)(*(_QWORD *)(v76 + 48) + 16LL * v75),
                                      *(_QWORD *)(*(_QWORD *)(v76 + 48) + 16LL * v75 + 8),
                                      v63,
                                      v81,
                                      (__int128)*a5);
        a5->m128i_i32[2] = v66;
        a6->m128i_i64[0] = 0;
        a6->m128i_i32[2] = 0;
        return a6;
      }
      return result;
  }
}
