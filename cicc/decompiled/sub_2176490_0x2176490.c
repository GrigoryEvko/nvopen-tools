// Function: sub_2176490
// Address: 0x2176490
//
__int64 __fastcall sub_2176490(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const void **a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int64 a11)
{
  __int64 v11; // r14
  unsigned int v12; // eax
  char v13; // r8
  unsigned int v14; // r10d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r11
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r10
  __int128 v21; // rax
  __m128i v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // [rsp+10h] [rbp-120h]
  __int64 v37; // [rsp+18h] [rbp-118h]
  __m128i v38; // [rsp+20h] [rbp-110h] BYREF
  __int64 v39; // [rsp+30h] [rbp-100h]
  __int64 v40; // [rsp+38h] [rbp-F8h]
  __int64 v41; // [rsp+40h] [rbp-F0h]
  __int64 v42; // [rsp+48h] [rbp-E8h]
  __int64 v43; // [rsp+50h] [rbp-E0h]
  __int64 v44; // [rsp+58h] [rbp-D8h]
  __int64 v45; // [rsp+60h] [rbp-D0h]
  __int64 v46; // [rsp+68h] [rbp-C8h]
  __int64 v47; // [rsp+70h] [rbp-C0h]
  __int64 v48; // [rsp+78h] [rbp-B8h]
  __int64 v49; // [rsp+80h] [rbp-B0h] BYREF
  const void **v50; // [rsp+88h] [rbp-A8h]
  __int128 v51; // [rsp+90h] [rbp-A0h]
  __int64 v52; // [rsp+A0h] [rbp-90h]
  _QWORD v53[4]; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v54[4]; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v55; // [rsp+F0h] [rbp-40h]

  v11 = a2;
  v38.m128i_i64[0] = a3;
  v40 = a4;
  v49 = a5;
  v50 = a6;
  if ( (_BYTE)a5 )
  {
    v12 = (unsigned int)sub_216FFF0(a5) >> 3;
    v14 = v12;
    if ( v13 == 9 )
    {
      LODWORD(v39) = v12;
      v15 = sub_1D309E0(a1, 158, a11, 5, 0, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a10);
      LOBYTE(v49) = 5;
      v14 = v39;
      v47 = v15;
      *(_QWORD *)&a10 = v15;
      v48 = v16;
      v50 = 0;
      *((_QWORD *)&a10 + 1) = (unsigned int)v16 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
    }
    else if ( v13 == 10 )
    {
      LODWORD(v39) = v12;
      v34 = sub_1D309E0(a1, 158, a11, 6, 0, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a10);
      LOBYTE(v49) = 6;
      v14 = v39;
      v45 = v34;
      *(_QWORD *)&a10 = v34;
      v46 = v35;
      v50 = 0;
      *((_QWORD *)&a10 + 1) = (unsigned int)v35 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
    }
  }
  else
  {
    v14 = (unsigned int)sub_1F58D40((__int64)&v49) >> 3;
  }
  if ( v14 )
  {
    v17 = a2;
    v18 = v38.m128i_i64[0];
    v19 = 8LL * v14;
    v20 = 0;
    v36 = v19;
    do
    {
      v39 = v20;
      v37 = v17;
      *(_QWORD *)&v21 = sub_1D38BB0((__int64)a1, v20, a11, 5, 0, 0, a7, a8, a9, 0);
      v22.m128i_i64[0] = (__int64)sub_1D332F0(
                                    a1,
                                    124,
                                    a11,
                                    (unsigned int)v49,
                                    v50,
                                    0,
                                    *(double *)a7.m128i_i64,
                                    a8,
                                    a9,
                                    a10,
                                    *((unsigned __int64 *)&a10 + 1),
                                    v21);
      v54[0] = v37;
      v54[1] = v18;
      v38 = v22;
      v23 = sub_1D38BB0((__int64)a1, v40, a11, 5, 0, 0, a7, a8, a9, 0);
      a7 = _mm_load_si128(&v38);
      v54[3] = v24;
      v55 = a7;
      v54[2] = v23;
      memset(v53, 0, 24);
      v51 = 0u;
      v52 = 0;
      v41 = 3;
      v42 = 0;
      v28 = sub_1D29190((__int64)a1, 1u, 0, v25, v26, v27);
      v30 = sub_1D251C0(a1, 675, a11, v28, v29, 0, v54, 3, v41, v42, v51, v52, 3u, 0, (__int64)v53);
      ++v40;
      v44 = v31;
      v17 = v30;
      v43 = v30;
      v20 = v39 + 8;
      v32 = (unsigned int)v31 | v18 & 0xFFFFFFFF00000000LL;
      v18 = v32;
    }
    while ( v39 + 8 != v36 );
    v38.m128i_i64[0] = v32;
    return v30;
  }
  return v11;
}
