// Function: sub_33EF7B0
// Address: 0x33ef7b0
//
__int64 __fastcall sub_33EF7B0(
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
  __m128i v18; // xmm1
  __m128i *v19; // rsi
  __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // r14
  __int64 (__fastcall *v23)(__int64, __int64, unsigned int); // rbx
  __int64 v24; // rax
  int v25; // edx
  unsigned __int16 v26; // ax
  __int64 v27; // r14
  __int64 v28; // rbx
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // rdi
  unsigned int v33; // r14d
  const __m128i *v34; // rax
  void (***v35)(); // rdi
  void (*v36)(); // rax
  _WORD *v37; // rsi
  __int64 v38; // r14
  __m128i v40; // xmm6
  __m128i v41; // xmm5
  __m128i *v42; // rsi
  __m128i *v43; // rax
  char v44; // [rsp+1Ch] [rbp-1254h]
  int v46; // [rsp+28h] [rbp-1248h]
  int v47; // [rsp+28h] [rbp-1248h]
  unsigned __int64 v48; // [rsp+A0h] [rbp-11D0h] BYREF
  __m128i *v49; // [rsp+A8h] [rbp-11C8h]
  const __m128i *v50; // [rsp+B0h] [rbp-11C0h]
  char v51[16]; // [rsp+C0h] [rbp-11B0h] BYREF
  __int64 v52; // [rsp+D0h] [rbp-11A0h]
  __m128i v53; // [rsp+E0h] [rbp-1190h] BYREF
  __m128i v54; // [rsp+F0h] [rbp-1180h] BYREF
  __m128i v55; // [rsp+100h] [rbp-1170h] BYREF
  __int64 v56; // [rsp+110h] [rbp-1160h] BYREF
  __int64 v57; // [rsp+118h] [rbp-1158h]
  __int64 v58; // [rsp+120h] [rbp-1150h]
  unsigned __int64 v59; // [rsp+128h] [rbp-1148h]
  __int64 v60; // [rsp+130h] [rbp-1140h]
  __int64 v61; // [rsp+138h] [rbp-1138h]
  __int64 v62; // [rsp+140h] [rbp-1130h]
  unsigned __int64 v63; // [rsp+148h] [rbp-1128h] BYREF
  __m128i *v64; // [rsp+150h] [rbp-1120h]
  const __m128i *v65; // [rsp+158h] [rbp-1118h]
  __int64 v66; // [rsp+160h] [rbp-1110h]
  __int64 v67; // [rsp+168h] [rbp-1108h] BYREF
  int v68; // [rsp+170h] [rbp-1100h]
  __int64 v69; // [rsp+178h] [rbp-10F8h]
  _BYTE *v70; // [rsp+180h] [rbp-10F0h]
  __int64 v71; // [rsp+188h] [rbp-10E8h]
  _BYTE v72[1792]; // [rsp+190h] [rbp-10E0h] BYREF
  _BYTE *v73; // [rsp+890h] [rbp-9E0h]
  __int64 v74; // [rsp+898h] [rbp-9D8h]
  _BYTE v75[512]; // [rsp+8A0h] [rbp-9D0h] BYREF
  _BYTE *v76; // [rsp+AA0h] [rbp-7D0h]
  __int64 v77; // [rsp+AA8h] [rbp-7C8h]
  _BYTE v78[1792]; // [rsp+AB0h] [rbp-7C0h] BYREF
  _BYTE *v79; // [rsp+11B0h] [rbp-C0h]
  __int64 v80; // [rsp+11B8h] [rbp-B8h]
  _BYTE v81[64]; // [rsp+11C0h] [rbp-B0h] BYREF
  __int64 v82; // [rsp+1200h] [rbp-70h]
  __int64 v83; // [rsp+1208h] [rbp-68h]
  int v84; // [rsp+1210h] [rbp-60h]
  char v85; // [rsp+1230h] [rbp-40h]

  v16 = *(__int64 **)(a1 + 40);
  v48 = 0;
  v44 = a11;
  v49 = 0;
  v50 = 0;
  v53 = 0u;
  v54 = 0u;
  v55 = 0u;
  v17 = sub_2E79000(v16);
  v54.m128i_i64[1] = sub_AE4420(v17, *(_QWORD *)(a1 + 64), 0);
  v54.m128i_i32[0] = a6;
  v53.m128i_i64[1] = a5;
  sub_332CDC0(&v48, 0, &v53);
  v41 = _mm_loadu_si128((const __m128i *)&a7);
  v42 = v49;
  v43 = (__m128i *)v50;
  v53.m128i_i64[1] = a7;
  v54.m128i_i32[0] = v41.m128i_i32[2];
  if ( v49 == v50 )
  {
    sub_332CDC0(&v48, v49, &v53);
    v40 = _mm_loadu_si128((const __m128i *)&a8);
    v19 = v49;
    v54.m128i_i64[1] = a9;
    v53.m128i_i64[1] = a8;
    v54.m128i_i32[0] = v40.m128i_i32[2];
    if ( v49 != v50 )
    {
      if ( !v49 )
        goto LABEL_4;
      goto LABEL_3;
    }
  }
  else
  {
    if ( v49 )
    {
      *v49 = _mm_loadu_si128(&v53);
      v42[1] = _mm_loadu_si128(&v54);
      v43 = (__m128i *)v50;
      v42[2] = _mm_loadu_si128(&v55);
      v42 = v49;
    }
    v18 = _mm_loadu_si128((const __m128i *)&a8);
    v19 = v42 + 3;
    v49 = v19;
    v53.m128i_i64[1] = a8;
    v54.m128i_i64[1] = a9;
    v54.m128i_i32[0] = v18.m128i_i32[2];
    if ( v19 != v43 )
    {
LABEL_3:
      *v19 = _mm_loadu_si128(&v53);
      v19[1] = _mm_loadu_si128(&v54);
      v19[2] = _mm_loadu_si128(&v55);
      v19 = v49;
LABEL_4:
      v49 = v19 + 3;
      goto LABEL_5;
    }
  }
  sub_332CDC0(&v48, v19, &v53);
LABEL_5:
  v46 = sub_2FE6650(a10);
  if ( v46 == 729 )
    sub_C64ED0("Unsupported element size", 1u);
  v20 = *(_QWORD *)a4;
  v59 = 0xFFFFFFFF00000020LL;
  v70 = v72;
  v71 = 0x2000000000LL;
  v74 = 0x2000000000LL;
  v77 = 0x2000000000LL;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = a1;
  v68 = 0;
  v69 = 0;
  v73 = v75;
  v76 = v78;
  v79 = v81;
  v80 = 0x400000000LL;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v67 = v20;
  if ( v20 )
    sub_B96E90((__int64)&v67, v20, 1);
  v21 = *(_DWORD *)(a4 + 8);
  v56 = a2;
  v68 = v21;
  v22 = *(_QWORD *)(a1 + 16);
  LODWORD(v57) = a3;
  v23 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v22 + 32LL);
  v24 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v23 == sub_2D42F30 )
  {
    v25 = sub_AE2980(v24, 0)[1];
    v26 = 2;
    if ( v25 != 1 )
    {
      v26 = 3;
      if ( v25 != 2 )
      {
        v26 = 4;
        if ( v25 != 4 )
        {
          v26 = 5;
          if ( v25 != 8 )
          {
            v26 = 6;
            if ( v25 != 16 )
            {
              v26 = 7;
              if ( v25 != 32 )
              {
                v26 = 8;
                if ( v25 != 64 )
                  v26 = 9 * (v25 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v26 = v23(v22, v24, 0);
  }
  v27 = v46;
  v28 = sub_33EED90(a1, *(const char **)(*(_QWORD *)(a1 + 16) + 8LL * v46 + 525288), v26, 0);
  v47 = v29;
  v30 = sub_BCB120(*(_QWORD **)(a1 + 64));
  v31 = *(_QWORD *)(a1 + 16);
  v58 = v30;
  v32 = v63;
  v33 = *(_DWORD *)(v31 + 4 * v27 + 531128);
  v61 = v28;
  LODWORD(v62) = v47;
  LODWORD(v60) = v33;
  v63 = v48;
  LODWORD(v30) = -1431655765 * ((__int64)((__int64)v49->m128i_i64 - v48) >> 4);
  v64 = v49;
  v48 = 0;
  v49 = 0;
  HIDWORD(v59) = v30;
  v34 = v50;
  v50 = 0;
  v65 = v34;
  if ( v32 )
    j_j___libc_free_0(v32);
  v35 = *(void (****)())(v66 + 16);
  v36 = **v35;
  if ( v36 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v36)(v35, *(_QWORD *)(v66 + 40), v33, &v63);
  v37 = *(_WORD **)(a1 + 16);
  LOBYTE(v59) = v59 & 0xDF;
  BYTE2(v59) = v44;
  sub_3377410((__int64)v51, v37, (__int64)&v56);
  v38 = v52;
  if ( v79 != v81 )
    _libc_free((unsigned __int64)v79);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v67 )
    sub_B91220((__int64)&v67, v67);
  if ( v63 )
    j_j___libc_free_0(v63);
  if ( v48 )
    j_j___libc_free_0(v48);
  return v38;
}
