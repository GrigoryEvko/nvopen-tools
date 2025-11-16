// Function: sub_1A43DA0
// Address: 0x1a43da0
//
__int64 __fastcall sub_1A43DA0(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // r15
  __int64 result; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // r8
  int v15; // r9d
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r8
  int v19; // r9d
  __int64 v20; // r8
  int v21; // r9d
  double v22; // xmm4_8
  double v23; // xmm5_8
  _BYTE *v24; // rdx
  _QWORD *v25; // rax
  _QWORD *i; // rdx
  unsigned __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r13
  __m128 *v30; // rdx
  char v31; // al
  _BYTE *v32; // r12
  _BYTE *v33; // r14
  _QWORD *v34; // r15
  __m128i v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 *v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rsi
  unsigned __int8 *v50; // rsi
  __int64 v51; // r14
  __m128i *v52; // rdx
  char v53; // al
  _BYTE *v54; // r12
  _BYTE *v55; // r15
  _BYTE *v56; // r13
  _QWORD *v57; // rbx
  __int64 v58; // rdx
  _QWORD *v59; // rax
  __int64 v60; // rdx
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 *v69; // r12
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rsi
  unsigned __int8 *v73; // rsi
  __int64 v74; // [rsp+28h] [rbp-328h]
  __int64 v75; // [rsp+30h] [rbp-320h]
  _QWORD *v76; // [rsp+30h] [rbp-320h]
  _QWORD *v77; // [rsp+38h] [rbp-318h]
  __int64 v78; // [rsp+38h] [rbp-318h]
  __int64 v79; // [rsp+40h] [rbp-310h]
  __int64 v80; // [rsp+60h] [rbp-2F0h]
  _QWORD *v81; // [rsp+68h] [rbp-2E8h]
  _QWORD *v82; // [rsp+68h] [rbp-2E8h]
  _QWORD v83[2]; // [rsp+70h] [rbp-2E0h] BYREF
  __m128i v84; // [rsp+80h] [rbp-2D0h] BYREF
  __int64 v85; // [rsp+90h] [rbp-2C0h]
  __m128i v86; // [rsp+A0h] [rbp-2B0h] BYREF
  __int64 v87; // [rsp+B0h] [rbp-2A0h]
  __m128 v88; // [rsp+C0h] [rbp-290h] BYREF
  __int64 v89; // [rsp+D0h] [rbp-280h]
  __m128 v90; // [rsp+E0h] [rbp-270h] BYREF
  __int64 v91; // [rsp+F0h] [rbp-260h]
  unsigned __int8 *v92; // [rsp+100h] [rbp-250h] BYREF
  __int64 v93; // [rsp+108h] [rbp-248h]
  unsigned __int64 *v94; // [rsp+110h] [rbp-240h]
  __int64 v95; // [rsp+118h] [rbp-238h]
  __int64 v96; // [rsp+120h] [rbp-230h]
  int v97; // [rsp+128h] [rbp-228h]
  __int64 v98; // [rsp+130h] [rbp-220h]
  __int64 v99; // [rsp+138h] [rbp-218h]
  _BYTE *v100; // [rsp+150h] [rbp-200h] BYREF
  __int64 v101; // [rsp+158h] [rbp-1F8h]
  _BYTE v102[64]; // [rsp+160h] [rbp-1F0h] BYREF
  __int64 v103[5]; // [rsp+1A0h] [rbp-1B0h] BYREF
  char *v104; // [rsp+1C8h] [rbp-188h]
  char v105; // [rsp+1D8h] [rbp-178h] BYREF
  __int64 v106[5]; // [rsp+220h] [rbp-130h] BYREF
  char *v107; // [rsp+248h] [rbp-108h]
  char v108; // [rsp+258h] [rbp-F8h] BYREF
  unsigned __int8 *v109[2]; // [rsp+2A0h] [rbp-B0h] BYREF
  __int16 v110; // [rsp+2B0h] [rbp-A0h]
  char *v111; // [rsp+2C8h] [rbp-88h]
  char v112; // [rsp+2D8h] [rbp-78h] BYREF

  v10 = (unsigned __int64)a2;
  if ( !*(_DWORD *)(a1 + 496) || (result = sub_1A3F5B0(a1, (__int64)a2), (_BYTE)result) )
  {
    result = 0;
    if ( *(_BYTE *)(*a2 + 8LL) == 16 )
    {
      v12 = *(_QWORD *)(*a2 + 32LL);
      v13 = sub_16498A0((__int64)a2);
      v16 = (unsigned __int8 *)a2[6];
      v92 = 0;
      v95 = v13;
      v17 = *(_QWORD *)(v10 + 40);
      v96 = 0;
      v93 = v17;
      v97 = 0;
      v98 = 0;
      v99 = 0;
      v94 = (unsigned __int64 *)(v10 + 24);
      v109[0] = v16;
      if ( v16 )
      {
        sub_1623A60((__int64)v109, (__int64)v16, 2);
        v92 = v109[0];
        if ( v109[0] )
          sub_1623210((__int64)v109, v109[0], (__int64)&v92);
      }
      sub_1A41500((__int64)v103, (_QWORD *)a1, v10, *(_QWORD *)(v10 - 48), v14, v15);
      sub_1A41500((__int64)v106, (_QWORD *)a1, v10, *(_QWORD *)(v10 - 24), v18, v19);
      v100 = v102;
      v101 = 0x800000000LL;
      v80 = (unsigned int)v12;
      if ( (_DWORD)v12 )
      {
        v24 = v102;
        v25 = v102;
        if ( (unsigned int)v12 > 8uLL )
        {
          sub_16CD150((__int64)&v100, v102, (unsigned int)v12, 8, v20, v21);
          v24 = v100;
          v25 = &v100[8 * (unsigned int)v101];
        }
        for ( i = &v24[8 * (unsigned int)v12]; i != v25; ++v25 )
        {
          if ( v25 )
            *v25 = 0;
        }
        LODWORD(v101) = v12;
      }
      v27 = *(_QWORD *)(v10 - 72);
      if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) == 16 )
      {
        v51 = 0;
        sub_1A41500((__int64)v109, (_QWORD *)a1, v10, v27, v20, v21);
        if ( (_DWORD)v12 )
        {
          v74 = v10;
          do
          {
            v86.m128i_i32[0] = v51;
            LOWORD(v87) = 265;
            v83[0] = sub_1649960(v74);
            v83[1] = v58;
            v84.m128i_i64[0] = (__int64)v83;
            LOWORD(v85) = 773;
            v84.m128i_i64[1] = (__int64)".i";
            v53 = v87;
            if ( (_BYTE)v87 )
            {
              if ( (_BYTE)v87 == 1 )
              {
                a4 = _mm_loadu_si128(&v84);
                v88 = (__m128)a4;
                v89 = v85;
              }
              else
              {
                v52 = (__m128i *)v86.m128i_i64[0];
                if ( BYTE1(v87) != 1 )
                {
                  v52 = &v86;
                  v53 = 2;
                }
                v88.m128_u64[1] = (unsigned __int64)v52;
                LOBYTE(v89) = 2;
                v88.m128_u64[0] = (unsigned __int64)&v84;
                BYTE1(v89) = v53;
              }
            }
            else
            {
              LOWORD(v89) = 256;
            }
            v54 = sub_1A3F820(v106, v51);
            v55 = sub_1A3F820(v103, v51);
            v56 = sub_1A3F820((__int64 *)v109, v51);
            v82 = &v100[8 * v51];
            if ( v56[16] > 0x10u || v55[16] > 0x10u || v54[16] > 0x10u )
            {
              LOWORD(v91) = 257;
              v59 = sub_1648A60(56, 3u);
              v57 = v59;
              if ( v59 )
              {
                v76 = v59 - 9;
                v78 = (__int64)v59;
                sub_15F1EA0((__int64)v59, *(_QWORD *)v55, 55, (__int64)(v59 - 9), 3, 0);
                if ( *(v57 - 9) )
                {
                  v60 = *(v57 - 8);
                  v61 = *(v57 - 7) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v61 = v60;
                  if ( v60 )
                    *(_QWORD *)(v60 + 16) = *(_QWORD *)(v60 + 16) & 3LL | v61;
                }
                *(v57 - 9) = v56;
                v62 = *((_QWORD *)v56 + 1);
                *(v57 - 8) = v62;
                if ( v62 )
                  *(_QWORD *)(v62 + 16) = (unsigned __int64)(v57 - 8) | *(_QWORD *)(v62 + 16) & 3LL;
                *(v57 - 7) = (unsigned __int64)(v56 + 8) | *(v57 - 7) & 3LL;
                *((_QWORD *)v56 + 1) = v76;
                if ( *(v57 - 6) )
                {
                  v63 = *(v57 - 5);
                  v64 = *(v57 - 4) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v64 = v63;
                  if ( v63 )
                    *(_QWORD *)(v63 + 16) = *(_QWORD *)(v63 + 16) & 3LL | v64;
                }
                *(v57 - 6) = v55;
                v65 = *((_QWORD *)v55 + 1);
                *(v57 - 5) = v65;
                if ( v65 )
                  *(_QWORD *)(v65 + 16) = (unsigned __int64)(v57 - 5) | *(_QWORD *)(v65 + 16) & 3LL;
                *(v57 - 4) = (unsigned __int64)(v55 + 8) | *(v57 - 4) & 3LL;
                *((_QWORD *)v55 + 1) = v57 - 6;
                if ( *(v57 - 3) )
                {
                  v66 = *(v57 - 2);
                  v67 = *(v57 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v67 = v66;
                  if ( v66 )
                    *(_QWORD *)(v66 + 16) = *(_QWORD *)(v66 + 16) & 3LL | v67;
                }
                *(v57 - 3) = v54;
                if ( v54 )
                {
                  v68 = *((_QWORD *)v54 + 1);
                  *(v57 - 2) = v68;
                  if ( v68 )
                    *(_QWORD *)(v68 + 16) = (unsigned __int64)(v57 - 2) | *(_QWORD *)(v68 + 16) & 3LL;
                  *(v57 - 1) = (unsigned __int64)(v54 + 8) | *(v57 - 1) & 3LL;
                  *((_QWORD *)v54 + 1) = v57 - 3;
                }
                sub_164B780((__int64)v57, (__int64 *)&v90);
              }
              else
              {
                v78 = 0;
              }
              if ( v93 )
              {
                v69 = v94;
                sub_157E9D0(v93 + 40, (__int64)v57);
                v70 = v57[3];
                v71 = *v69;
                v57[4] = v69;
                v71 &= 0xFFFFFFFFFFFFFFF8LL;
                v57[3] = v71 | v70 & 7;
                *(_QWORD *)(v71 + 8) = v57 + 3;
                *v69 = *v69 & 7 | (unsigned __int64)(v57 + 3);
              }
              sub_164B780(v78, (__int64 *)&v88);
              if ( v92 )
              {
                v90.m128_u64[0] = (unsigned __int64)v92;
                sub_1623A60((__int64)&v90, (__int64)v92, 2);
                v72 = v57[6];
                if ( v72 )
                  sub_161E7C0((__int64)(v57 + 6), v72);
                v73 = (unsigned __int8 *)v90.m128_u64[0];
                v57[6] = v90.m128_u64[0];
                if ( v73 )
                  sub_1623210((__int64)&v90, v73, (__int64)(v57 + 6));
              }
            }
            else
            {
              v57 = (_QWORD *)sub_15A2DC0((__int64)v56, (__int64 *)v55, (__int64)v54, 0);
            }
            ++v51;
            *v82 = v57;
          }
          while ( v80 != v51 );
          v10 = v74;
        }
        if ( v111 != &v112 )
          _libc_free((unsigned __int64)v111);
      }
      else
      {
        v28 = 0;
        if ( (_DWORD)v12 )
        {
          v75 = v10;
          v29 = *(_QWORD *)(v10 - 72);
          do
          {
            v88.m128_i32[0] = v28;
            LOWORD(v89) = 265;
            v35.m128i_i64[0] = (__int64)sub_1649960(v75);
            v84 = v35;
            v86.m128i_i64[0] = (__int64)&v84;
            LOWORD(v87) = 773;
            v86.m128i_i64[1] = (__int64)".i";
            v31 = v89;
            if ( (_BYTE)v89 )
            {
              if ( (_BYTE)v89 == 1 )
              {
                a3 = (__m128)_mm_loadu_si128(&v86);
                v90 = a3;
                v91 = v87;
              }
              else
              {
                v30 = (__m128 *)v88.m128_u64[0];
                if ( BYTE1(v89) != 1 )
                {
                  v30 = &v88;
                  v31 = 2;
                }
                v90.m128_u64[1] = (unsigned __int64)v30;
                LOBYTE(v91) = 2;
                v90.m128_u64[0] = (unsigned __int64)&v86;
                BYTE1(v91) = v31;
              }
            }
            else
            {
              LOWORD(v91) = 256;
            }
            v32 = sub_1A3F820(v106, v28);
            v33 = sub_1A3F820(v103, v28);
            v81 = &v100[8 * v28];
            if ( *(_BYTE *)(v29 + 16) > 0x10u || v33[16] > 0x10u || v32[16] > 0x10u )
            {
              v110 = 257;
              v36 = sub_1648A60(56, 3u);
              v34 = v36;
              if ( v36 )
              {
                v77 = v36 - 9;
                v79 = (__int64)v36;
                sub_15F1EA0((__int64)v36, *(_QWORD *)v33, 55, (__int64)(v36 - 9), 3, 0);
                if ( *(v34 - 9) )
                {
                  v37 = *(v34 - 8);
                  v38 = *(v34 - 7) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v38 = v37;
                  if ( v37 )
                    *(_QWORD *)(v37 + 16) = *(_QWORD *)(v37 + 16) & 3LL | v38;
                }
                *(v34 - 9) = v29;
                v39 = *(_QWORD *)(v29 + 8);
                *(v34 - 8) = v39;
                if ( v39 )
                  *(_QWORD *)(v39 + 16) = (unsigned __int64)(v34 - 8) | *(_QWORD *)(v39 + 16) & 3LL;
                *(v34 - 7) = (v29 + 8) | *(v34 - 7) & 3LL;
                *(_QWORD *)(v29 + 8) = v77;
                if ( *(v34 - 6) )
                {
                  v40 = *(v34 - 5);
                  v41 = *(v34 - 4) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v41 = v40;
                  if ( v40 )
                    *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
                }
                *(v34 - 6) = v33;
                v42 = *((_QWORD *)v33 + 1);
                *(v34 - 5) = v42;
                if ( v42 )
                  *(_QWORD *)(v42 + 16) = (unsigned __int64)(v34 - 5) | *(_QWORD *)(v42 + 16) & 3LL;
                *(v34 - 4) = (unsigned __int64)(v33 + 8) | *(v34 - 4) & 3LL;
                *((_QWORD *)v33 + 1) = v34 - 6;
                if ( *(v34 - 3) )
                {
                  v43 = *(v34 - 2);
                  v44 = *(v34 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v44 = v43;
                  if ( v43 )
                    *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v44;
                }
                *(v34 - 3) = v32;
                if ( v32 )
                {
                  v45 = *((_QWORD *)v32 + 1);
                  *(v34 - 2) = v45;
                  if ( v45 )
                    *(_QWORD *)(v45 + 16) = (unsigned __int64)(v34 - 2) | *(_QWORD *)(v45 + 16) & 3LL;
                  *(v34 - 1) = (unsigned __int64)(v32 + 8) | *(v34 - 1) & 3LL;
                  *((_QWORD *)v32 + 1) = v34 - 3;
                }
                sub_164B780((__int64)v34, (__int64 *)v109);
              }
              else
              {
                v79 = 0;
              }
              if ( v93 )
              {
                v46 = v94;
                sub_157E9D0(v93 + 40, (__int64)v34);
                v47 = v34[3];
                v48 = *v46;
                v34[4] = v46;
                v48 &= 0xFFFFFFFFFFFFFFF8LL;
                v34[3] = v48 | v47 & 7;
                *(_QWORD *)(v48 + 8) = v34 + 3;
                *v46 = *v46 & 7 | (unsigned __int64)(v34 + 3);
              }
              sub_164B780(v79, (__int64 *)&v90);
              if ( v92 )
              {
                v109[0] = v92;
                sub_1623A60((__int64)v109, (__int64)v92, 2);
                v49 = v34[6];
                if ( v49 )
                  sub_161E7C0((__int64)(v34 + 6), v49);
                v50 = v109[0];
                v34[6] = v109[0];
                if ( v50 )
                  sub_1623210((__int64)v109, v50, (__int64)(v34 + 6));
              }
            }
            else
            {
              v34 = (_QWORD *)sub_15A2DC0(v29, (__int64 *)v33, (__int64)v32, 0);
            }
            ++v28;
            *v81 = v34;
          }
          while ( v80 != v28 );
          v10 = v75;
        }
      }
      sub_1A41120(a1, v10, &v100, a3, *(double *)a4.m128i_i64, a5, a6, v22, v23, a9, a10);
      if ( v100 != v102 )
        _libc_free((unsigned __int64)v100);
      if ( v107 != &v108 )
        _libc_free((unsigned __int64)v107);
      if ( v104 != &v105 )
        _libc_free((unsigned __int64)v104);
      result = 1;
      if ( v92 )
      {
        sub_161E7C0((__int64)&v92, (__int64)v92);
        return 1;
      }
    }
  }
  return result;
}
