// Function: sub_3493EB0
// Address: 0x3493eb0
//
__int64 __fastcall sub_3493EB0(_WORD *a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int); // r14
  __int64 v7; // rax
  int v8; // edx
  unsigned __int16 v9; // ax
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // eax
  unsigned __int8 *v14; // rdi
  unsigned __int8 *v15; // rax
  unsigned __int8 *v16; // r13
  const char *v17; // rax
  __int64 v18; // r9
  size_t v19; // rdx
  size_t v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdi
  _BYTE *v24; // rax
  _QWORD *v25; // rax
  __m128i *v26; // rsi
  __int32 v27; // edx
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // r13
  int v31; // ecx
  unsigned __int64 v32; // rax
  int v33; // edx
  const __m128i *v34; // rdx
  void (***v35)(); // rdi
  void (*v36)(); // rax
  __int64 v37; // r12
  int v39; // [rsp+0h] [rbp-1270h]
  size_t v40; // [rsp+8h] [rbp-1268h]
  const char *v41; // [rsp+10h] [rbp-1260h]
  size_t v42; // [rsp+18h] [rbp-1258h]
  __int64 v43; // [rsp+28h] [rbp-1248h]
  __int64 v44; // [rsp+50h] [rbp-1220h] BYREF
  int v45; // [rsp+58h] [rbp-1218h]
  unsigned __int64 v46; // [rsp+60h] [rbp-1210h] BYREF
  __m128i *v47; // [rsp+68h] [rbp-1208h]
  const __m128i *v48; // [rsp+70h] [rbp-1200h]
  __int64 v49; // [rsp+80h] [rbp-11F0h] BYREF
  __m128i v50; // [rsp+A0h] [rbp-11D0h] BYREF
  __m128i v51; // [rsp+B0h] [rbp-11C0h] BYREF
  __m128i v52; // [rsp+C0h] [rbp-11B0h] BYREF
  _QWORD *v53; // [rsp+D0h] [rbp-11A0h] BYREF
  __int64 v54; // [rsp+D8h] [rbp-1198h]
  unsigned __int64 v55; // [rsp+E0h] [rbp-1190h]
  _QWORD v56[5]; // [rsp+E8h] [rbp-1188h] BYREF
  __int64 v57; // [rsp+110h] [rbp-1160h] BYREF
  __int64 v58; // [rsp+118h] [rbp-1158h]
  __int64 v59; // [rsp+120h] [rbp-1150h]
  unsigned __int64 v60; // [rsp+128h] [rbp-1148h]
  __int64 v61; // [rsp+130h] [rbp-1140h]
  __int64 v62; // [rsp+138h] [rbp-1138h]
  __int64 v63; // [rsp+140h] [rbp-1130h]
  unsigned __int64 v64; // [rsp+148h] [rbp-1128h] BYREF
  __m128i *v65; // [rsp+150h] [rbp-1120h]
  const __m128i *v66; // [rsp+158h] [rbp-1118h]
  __int64 v67; // [rsp+160h] [rbp-1110h]
  __int64 v68; // [rsp+168h] [rbp-1108h] BYREF
  int v69; // [rsp+170h] [rbp-1100h]
  __int64 v70; // [rsp+178h] [rbp-10F8h]
  _BYTE *v71; // [rsp+180h] [rbp-10F0h]
  __int64 v72; // [rsp+188h] [rbp-10E8h]
  _BYTE v73[1792]; // [rsp+190h] [rbp-10E0h] BYREF
  _BYTE *v74; // [rsp+890h] [rbp-9E0h]
  __int64 v75; // [rsp+898h] [rbp-9D8h]
  _BYTE v76[512]; // [rsp+8A0h] [rbp-9D0h] BYREF
  _BYTE *v77; // [rsp+AA0h] [rbp-7D0h]
  __int64 v78; // [rsp+AA8h] [rbp-7C8h]
  _BYTE v79[1792]; // [rsp+AB0h] [rbp-7C0h] BYREF
  _BYTE *v80; // [rsp+11B0h] [rbp-C0h]
  __int64 v81; // [rsp+11B8h] [rbp-B8h]
  _BYTE v82[64]; // [rsp+11C0h] [rbp-B0h] BYREF
  __int64 v83; // [rsp+1200h] [rbp-70h]
  __int64 v84; // [rsp+1208h] [rbp-68h]
  int v85; // [rsp+1210h] [rbp-60h]
  char v86; // [rsp+1230h] [rbp-40h]

  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  v7 = sub_2E79000(*(__int64 **)(a3 + 40));
  if ( v6 == sub_2D42F30 )
  {
    v8 = sub_AE2980(v7, 0)[1];
    v9 = 2;
    if ( v8 != 1 )
    {
      v9 = 3;
      if ( v8 != 2 )
      {
        v9 = 4;
        if ( v8 != 4 )
        {
          v9 = 5;
          if ( v8 != 8 )
          {
            v9 = 6;
            if ( v8 != 16 )
            {
              v9 = 7;
              if ( v8 != 32 )
              {
                v9 = 8;
                if ( v8 != 64 )
                  v9 = 9 * (v8 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v9 = v6((__int64)a1, v7, 0);
  }
  v10 = v9;
  v11 = sub_BCE3C0(*(__int64 **)(a3 + 64), 0);
  v12 = *(_QWORD *)(a2 + 80);
  v43 = v11;
  v44 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v44, v12, 1);
  v13 = *(_DWORD *)(a2 + 72);
  v14 = *(unsigned __int8 **)(a2 + 96);
  v46 = 0;
  v47 = 0;
  v45 = v13;
  v48 = 0;
  v50 = 0u;
  v51 = 0u;
  v52 = 0u;
  v15 = sub_BD3BE0(v14, v12);
  v55 = 32;
  v16 = v15;
  v53 = v56;
  qmemcpy(v56, "__emutls_v.", 11);
  v54 = 11;
  v17 = sub_BD5D20((__int64)v15);
  v20 = v19;
  v21 = v54;
  if ( v20 + v54 > v55 )
  {
    v40 = v20;
    v41 = v17;
    sub_C8D290((__int64)&v53, v56, v20 + v54, 1u, v20, v18);
    v21 = v54;
    v20 = v40;
    v17 = v41;
  }
  v22 = (__int64)v53;
  if ( v20 )
  {
    v42 = v20;
    memcpy((char *)v53 + v21, v17, v20);
    v22 = (__int64)v53;
    v21 = v54;
    v20 = v42;
  }
  v23 = *((_QWORD *)v16 + 5);
  v54 = v20 + v21;
  v24 = sub_BA8CD0(v23, v22, v20 + v21, 1);
  v25 = sub_33ED290(a3, (unsigned __int64)v24, (__int64)&v44, v10, 0, 0, 0, 0);
  v26 = v47;
  v50.m128i_i64[1] = (__int64)v25;
  v51.m128i_i32[0] = v27;
  v51.m128i_i64[1] = v43;
  if ( v47 == v48 )
  {
    sub_332CDC0(&v46, v47, &v50);
  }
  else
  {
    if ( v47 )
    {
      *v47 = _mm_loadu_si128(&v50);
      v26[1] = _mm_loadu_si128(&v51);
      v26[2] = _mm_loadu_si128(&v52);
      v26 = v47;
    }
    v47 = v26 + 3;
  }
  v28 = sub_33EED90(a3, "__emutls_get_address", v10, 0);
  v57 = 0;
  v30 = v28;
  v31 = v29;
  v60 = 0xFFFFFFFF00000020LL;
  v71 = v73;
  v72 = 0x2000000000LL;
  v75 = 0x2000000000LL;
  v78 = 0x2000000000LL;
  v81 = 0x400000000LL;
  v32 = v44;
  v74 = v76;
  v58 = 0;
  v59 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = a3;
  v69 = 0;
  v70 = 0;
  v77 = v79;
  v80 = v82;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v68 = v44;
  if ( v44 )
  {
    v39 = v29;
    sub_B96E90((__int64)&v68, v44, 1);
    v32 = v64;
    v31 = v39;
  }
  LODWORD(v58) = 0;
  v69 = v45;
  v57 = a3 + 288;
  v59 = v43;
  v62 = v30;
  LODWORD(v63) = v31;
  v64 = v46;
  LODWORD(v61) = 0;
  v33 = -1431655765 * ((__int64)((__int64)v47->m128i_i64 - v46) >> 4);
  v65 = v47;
  v46 = 0;
  v47 = 0;
  HIDWORD(v60) = v33;
  v34 = v48;
  v48 = 0;
  v66 = v34;
  if ( v32 )
    j_j___libc_free_0(v32);
  v35 = *(void (****)())(v67 + 16);
  v36 = **v35;
  if ( v36 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v36)(v35, *(_QWORD *)(v67 + 40), 0, &v64);
  sub_3377410((__int64)&v49, a1, (__int64)&v57);
  v37 = v49;
  *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL) + 65LL) = 257;
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
  if ( v53 != v56 )
    _libc_free((unsigned __int64)v53);
  if ( v46 )
    j_j___libc_free_0(v46);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v37;
}
