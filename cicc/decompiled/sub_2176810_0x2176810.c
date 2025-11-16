// Function: sub_2176810
// Address: 0x2176810
//
__int64 __fastcall sub_2176810(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned int a10,
        const void **a11,
        __int128 a12,
        __int64 a13,
        unsigned __int64 a14,
        __int64 a15)
{
  __int64 v16; // r14
  int v17; // edx
  unsigned int v18; // eax
  char v19; // cl
  __int64 v20; // r10
  unsigned int v21; // r11d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r11
  __int128 v26; // rax
  __m128i v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // [rsp+10h] [rbp-170h]
  int v40; // [rsp+28h] [rbp-158h]
  __int64 v41; // [rsp+30h] [rbp-150h]
  unsigned int v42; // [rsp+38h] [rbp-148h]
  __int64 v43; // [rsp+38h] [rbp-148h]
  unsigned int v44; // [rsp+38h] [rbp-148h]
  __int64 v45; // [rsp+40h] [rbp-140h]
  __int64 v46; // [rsp+40h] [rbp-140h]
  __int64 v48; // [rsp+48h] [rbp-138h]
  __m128i v49; // [rsp+50h] [rbp-130h] BYREF
  __int64 v50; // [rsp+60h] [rbp-120h]
  unsigned __int64 v51; // [rsp+68h] [rbp-118h]
  unsigned int v52; // [rsp+70h] [rbp-110h]
  unsigned int v53; // [rsp+74h] [rbp-10Ch]
  unsigned __int64 v54; // [rsp+78h] [rbp-108h]
  __int64 v55; // [rsp+80h] [rbp-100h]
  __int64 v56; // [rsp+88h] [rbp-F8h]
  __int64 v57; // [rsp+90h] [rbp-F0h]
  __int64 v58; // [rsp+98h] [rbp-E8h]
  __int64 v59; // [rsp+A0h] [rbp-E0h]
  __int64 v60; // [rsp+A8h] [rbp-D8h]
  __int64 v61; // [rsp+B0h] [rbp-D0h]
  __int64 v62; // [rsp+B8h] [rbp-C8h]
  __int64 v63; // [rsp+C8h] [rbp-B8h]
  __int64 v64; // [rsp+D0h] [rbp-B0h]
  _QWORD v65[4]; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v66[6]; // [rsp+100h] [rbp-80h] BYREF
  __m128i v67; // [rsp+130h] [rbp-50h]
  __int64 v68; // [rsp+140h] [rbp-40h]
  unsigned __int64 v69; // [rsp+148h] [rbp-38h]

  v54 = a3;
  v52 = a5;
  v16 = a15;
  v49.m128i_i64[0] = a2;
  v50 = a13;
  v51 = a14;
  v53 = a14;
  v45 = sub_1D252B0((__int64)a1, 1, 0, 111, 0);
  v40 = v17;
  if ( (_BYTE)a10 )
  {
    v18 = (unsigned int)sub_216FFF0(a10) >> 3;
    v21 = v18;
    if ( v19 == 9 )
    {
      v49.m128i_i64[0] = v20;
      v42 = v18;
      v22 = sub_1D309E0(a1, 158, v16, 5, 0, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a12);
      LOBYTE(a10) = 5;
      v21 = v42;
      v61 = v22;
      v20 = v49.m128i_i64[0];
      *(_QWORD *)&a12 = v22;
      v62 = v23;
      a11 = 0;
      *((_QWORD *)&a12 + 1) = (unsigned int)v23 | *((_QWORD *)&a12 + 1) & 0xFFFFFFFF00000000LL;
    }
    else if ( v19 == 10 )
    {
      v49.m128i_i64[0] = v20;
      v44 = v18;
      v37 = sub_1D309E0(a1, 158, v16, 6, 0, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a12);
      LOBYTE(a10) = 6;
      v20 = v49.m128i_i64[0];
      v59 = v37;
      v21 = v44;
      *(_QWORD *)&a12 = v37;
      v60 = v38;
      a11 = 0;
      *((_QWORD *)&a12 + 1) = (unsigned int)v38 | *((_QWORD *)&a12 + 1) & 0xFFFFFFFF00000000LL;
    }
  }
  else
  {
    v36 = sub_1F58D40((__int64)&a10);
    v20 = v49.m128i_i64[0];
    v21 = v36 >> 3;
  }
  if ( v21 )
  {
    v41 = a6;
    v24 = v21;
    v25 = 0;
    v43 = 8 * v24;
    v39 = v45;
    while ( 1 )
    {
      v48 = v25;
      v46 = v20;
      *(_QWORD *)&v26 = sub_1D38BB0((__int64)a1, v25, v16, 5, 0, 0, a7, a8, a9, 0);
      v27.m128i_i64[0] = (__int64)sub_1D332F0(
                                    a1,
                                    124,
                                    v16,
                                    a10,
                                    a11,
                                    0,
                                    *(double *)a7.m128i_i64,
                                    a8,
                                    a9,
                                    a12,
                                    *((unsigned __int64 *)&a12 + 1),
                                    v26);
      v49 = v27;
      v66[0] = v46;
      v66[1] = v54;
      v28 = sub_1D38BB0((__int64)a1, v41, v16, 5, 0, 0, a7, a8, a9, 0);
      v29 = a4;
      v66[3] = v30;
      ++a4;
      v66[2] = v28;
      v31 = sub_1D38BB0((__int64)a1, v29, v16, 5, 0, 0, a7, a8, a9, 0);
      a7 = _mm_load_si128(&v49);
      v66[4] = v31;
      v66[5] = v32;
      v51 = v53 | v51 & 0xFFFFFFFF00000000LL;
      v68 = v50;
      v69 = v51;
      v63 = 0;
      v64 = 0;
      v55 = 3;
      v56 = 0;
      v67 = a7;
      memset(v65, 0, 24);
      v33 = sub_1D251C0(a1, 670, v16, v39, v40, v52, v66, 5, 3, 0, 0, 0, 3u, 0, (__int64)v65);
      v53 = 1;
      v57 = v33;
      v20 = v33;
      v58 = v34;
      v25 = v48 + 8;
      v54 = (unsigned int)v34 | v54 & 0xFFFFFFFF00000000LL;
      if ( v48 + 8 == v43 )
        break;
      v50 = v33;
    }
  }
  return v20;
}
