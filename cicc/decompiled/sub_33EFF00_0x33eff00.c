// Function: sub_33EFF00
// Address: 0x33eff00
//
__int64 __fastcall sub_33EFF00(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int32 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        unsigned int a10,
        char a11)
{
  __int64 *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm0
  __m128i *v20; // rsi
  __m128i *v21; // rax
  __m128i v22; // xmm1
  __m128i *v23; // rsi
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // r14
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int); // rbx
  __int64 v28; // rax
  int v29; // edx
  unsigned __int16 v30; // ax
  __int64 v31; // r14
  __int64 v32; // rbx
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // rdi
  unsigned int v37; // r14d
  const __m128i *v38; // rax
  void (***v39)(); // rdi
  void (*v40)(); // rax
  _WORD *v41; // rsi
  __int64 v42; // r14
  __m128i v44; // xmm5
  char v45; // [rsp+1Ch] [rbp-1244h]
  int v47; // [rsp+28h] [rbp-1238h]
  int v48; // [rsp+28h] [rbp-1238h]
  unsigned __int64 v49; // [rsp+90h] [rbp-11D0h] BYREF
  __m128i *v50; // [rsp+98h] [rbp-11C8h]
  const __m128i *v51; // [rsp+A0h] [rbp-11C0h]
  char v52[16]; // [rsp+B0h] [rbp-11B0h] BYREF
  __int64 v53; // [rsp+C0h] [rbp-11A0h]
  __m128i v54; // [rsp+D0h] [rbp-1190h] BYREF
  __m128i v55; // [rsp+E0h] [rbp-1180h] BYREF
  __m128i v56; // [rsp+F0h] [rbp-1170h] BYREF
  __int64 v57; // [rsp+100h] [rbp-1160h] BYREF
  __int64 v58; // [rsp+108h] [rbp-1158h]
  __int64 v59; // [rsp+110h] [rbp-1150h]
  unsigned __int64 v60; // [rsp+118h] [rbp-1148h]
  __int64 v61; // [rsp+120h] [rbp-1140h]
  __int64 v62; // [rsp+128h] [rbp-1138h]
  __int64 v63; // [rsp+130h] [rbp-1130h]
  unsigned __int64 v64; // [rsp+138h] [rbp-1128h] BYREF
  __m128i *v65; // [rsp+140h] [rbp-1120h]
  const __m128i *v66; // [rsp+148h] [rbp-1118h]
  __int64 v67; // [rsp+150h] [rbp-1110h]
  __int64 v68; // [rsp+158h] [rbp-1108h] BYREF
  int v69; // [rsp+160h] [rbp-1100h]
  __int64 v70; // [rsp+168h] [rbp-10F8h]
  _BYTE *v71; // [rsp+170h] [rbp-10F0h]
  __int64 v72; // [rsp+178h] [rbp-10E8h]
  _BYTE v73[1792]; // [rsp+180h] [rbp-10E0h] BYREF
  _BYTE *v74; // [rsp+880h] [rbp-9E0h]
  __int64 v75; // [rsp+888h] [rbp-9D8h]
  _BYTE v76[512]; // [rsp+890h] [rbp-9D0h] BYREF
  _BYTE *v77; // [rsp+A90h] [rbp-7D0h]
  __int64 v78; // [rsp+A98h] [rbp-7C8h]
  _BYTE v79[1792]; // [rsp+AA0h] [rbp-7C0h] BYREF
  _BYTE *v80; // [rsp+11A0h] [rbp-C0h]
  __int64 v81; // [rsp+11A8h] [rbp-B8h]
  _BYTE v82[64]; // [rsp+11B0h] [rbp-B0h] BYREF
  __int64 v83; // [rsp+11F0h] [rbp-70h]
  __int64 v84; // [rsp+11F8h] [rbp-68h]
  int v85; // [rsp+1200h] [rbp-60h]
  char v86; // [rsp+1220h] [rbp-40h]

  v16 = *(__int64 **)(a1 + 40);
  v49 = 0;
  v45 = a11;
  v50 = 0;
  v51 = 0;
  v54 = 0u;
  v55 = 0u;
  v56 = 0u;
  v17 = sub_2E79000(v16);
  v55.m128i_i64[1] = sub_AE4420(v17, *(_QWORD *)(a1 + 64), 0);
  v54.m128i_i64[1] = a5;
  v55.m128i_i32[0] = a6;
  sub_332CDC0(&v49, 0, &v54);
  v18 = sub_BCB2B0(*(_QWORD **)(a1 + 64));
  v19 = _mm_loadu_si128((const __m128i *)&a7);
  v20 = v50;
  v55.m128i_i64[1] = v18;
  v54.m128i_i64[1] = a7;
  v55.m128i_i32[0] = v19.m128i_i32[2];
  v21 = (__m128i *)v51;
  if ( v50 == v51 )
  {
    sub_332CDC0(&v49, v50, &v54);
    v44 = _mm_loadu_si128((const __m128i *)&a8);
    v23 = v50;
    v55.m128i_i64[1] = a9;
    v54.m128i_i64[1] = a8;
    v55.m128i_i32[0] = v44.m128i_i32[2];
    if ( v50 != v51 )
    {
      if ( !v50 )
        goto LABEL_6;
      goto LABEL_5;
    }
  }
  else
  {
    if ( v50 )
    {
      *v50 = _mm_loadu_si128(&v54);
      v20[1] = _mm_loadu_si128(&v55);
      v21 = (__m128i *)v51;
      v20[2] = _mm_loadu_si128(&v56);
      v20 = v50;
    }
    v22 = _mm_loadu_si128((const __m128i *)&a8);
    v23 = v20 + 3;
    v50 = v23;
    v54.m128i_i64[1] = a8;
    v55.m128i_i64[1] = a9;
    v55.m128i_i32[0] = v22.m128i_i32[2];
    if ( v23 != v21 )
    {
LABEL_5:
      *v23 = _mm_loadu_si128(&v54);
      v23[1] = _mm_loadu_si128(&v55);
      v23[2] = _mm_loadu_si128(&v56);
      v23 = v50;
LABEL_6:
      v50 = v23 + 3;
      goto LABEL_7;
    }
  }
  sub_332CDC0(&v49, v23, &v54);
LABEL_7:
  v47 = sub_2FE6670(a10);
  if ( v47 == 729 )
    sub_C64ED0("Unsupported element size", 1u);
  v24 = *(_QWORD *)a4;
  v60 = 0xFFFFFFFF00000020LL;
  v71 = v73;
  v72 = 0x2000000000LL;
  v75 = 0x2000000000LL;
  v78 = 0x2000000000LL;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = a1;
  v69 = 0;
  v70 = 0;
  v74 = v76;
  v77 = v79;
  v80 = v82;
  v81 = 0x400000000LL;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v68 = v24;
  if ( v24 )
    sub_B96E90((__int64)&v68, v24, 1);
  v25 = *(_DWORD *)(a4 + 8);
  v57 = a2;
  v69 = v25;
  v26 = *(_QWORD *)(a1 + 16);
  LODWORD(v58) = a3;
  v27 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v26 + 32LL);
  v28 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v27 == sub_2D42F30 )
  {
    v29 = sub_AE2980(v28, 0)[1];
    v30 = 2;
    if ( v29 != 1 )
    {
      v30 = 3;
      if ( v29 != 2 )
      {
        v30 = 4;
        if ( v29 != 4 )
        {
          v30 = 5;
          if ( v29 != 8 )
          {
            v30 = 6;
            if ( v29 != 16 )
            {
              v30 = 7;
              if ( v29 != 32 )
              {
                v30 = 8;
                if ( v29 != 64 )
                  v30 = 9 * (v29 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v30 = v27(v26, v28, 0);
  }
  v31 = v47;
  v32 = sub_33EED90(a1, *(const char **)(*(_QWORD *)(a1 + 16) + 8LL * v47 + 525288), v30, 0);
  v48 = v33;
  v34 = sub_BCB120(*(_QWORD **)(a1 + 64));
  v35 = *(_QWORD *)(a1 + 16);
  v59 = v34;
  v36 = v64;
  v37 = *(_DWORD *)(v35 + 4 * v31 + 531128);
  v62 = v32;
  LODWORD(v63) = v48;
  LODWORD(v61) = v37;
  v64 = v49;
  LODWORD(v34) = -1431655765 * ((__int64)((__int64)v50->m128i_i64 - v49) >> 4);
  v65 = v50;
  v49 = 0;
  v50 = 0;
  HIDWORD(v60) = v34;
  v38 = v51;
  v51 = 0;
  v66 = v38;
  if ( v36 )
    j_j___libc_free_0(v36);
  v39 = *(void (****)())(v67 + 16);
  v40 = **v39;
  if ( v40 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v40)(v39, *(_QWORD *)(v67 + 40), v37, &v64);
  v41 = *(_WORD **)(a1 + 16);
  LOBYTE(v60) = v60 & 0xDF;
  BYTE2(v60) = v45;
  sub_3377410((__int64)v52, v41, (__int64)&v57);
  v42 = v53;
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
  if ( v64 )
    j_j___libc_free_0(v64);
  if ( v49 )
    j_j___libc_free_0(v49);
  return v42;
}
