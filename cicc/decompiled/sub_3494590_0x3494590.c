// Function: sub_3494590
// Address: 0x3494590
//
__int64 __fastcall sub_3494590(
        __int64 a1,
        _WORD *a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 (__fastcall *a6)(__int64 a1, __int64 a2, unsigned int a3),
        __int64 a7,
        unsigned __int64 a8,
        __int64 a9,
        int a10,
        unsigned int a11,
        __int64 a12,
        char a13,
        __int64 a14,
        __int64 a15,
        __int64 a16)
{
  _WORD *v16; // r13
  unsigned int v18; // ebx
  __int64 v19; // rcx
  __int64 (*v21)(); // rax
  __m128i *v22; // rsi
  __int64 v23; // r13
  __int64 *v24; // rsi
  unsigned __int16 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  char v28; // al
  __int64 (__fastcall *v29)(__int64, __int64, unsigned int); // rbx
  __int64 v30; // rax
  int v31; // edx
  unsigned __int16 v32; // ax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int); // rcx
  __int64 v40; // rsi
  unsigned int v41; // ebx
  __m128i *v42; // rdx
  int v43; // eax
  const __m128i *v44; // rcx
  unsigned __int64 v45; // rdi
  const __m128i *v46; // rax
  void (***v47)(); // rdi
  void (*v48)(); // rax
  __int64 (*v50)(); // rax
  char v51; // al
  char v52; // cl
  unsigned __int8 v53; // cl
  int v54; // [rsp+0h] [rbp-1220h]
  __int64 v55; // [rsp+8h] [rbp-1218h]
  __int64 v56; // [rsp+10h] [rbp-1210h]
  char v59; // [rsp+28h] [rbp-11F8h]
  __int64 v60; // [rsp+30h] [rbp-11F0h]
  int v61; // [rsp+38h] [rbp-11E8h]
  unsigned __int8 v62; // [rsp+3Fh] [rbp-11E1h]
  __m128i v63; // [rsp+50h] [rbp-11D0h]
  _QWORD v64[2]; // [rsp+60h] [rbp-11C0h] BYREF
  const __m128i *v65; // [rsp+70h] [rbp-11B0h] BYREF
  __m128i *v66; // [rsp+78h] [rbp-11A8h]
  const __m128i *v67; // [rsp+80h] [rbp-11A0h]
  __m128i v68; // [rsp+90h] [rbp-1190h] BYREF
  __m128i v69; // [rsp+A0h] [rbp-1180h] BYREF
  __m128i v70; // [rsp+B0h] [rbp-1170h] BYREF
  __int64 v71; // [rsp+C0h] [rbp-1160h] BYREF
  __int64 v72; // [rsp+C8h] [rbp-1158h]
  __int64 v73; // [rsp+D0h] [rbp-1150h]
  unsigned __int64 v74; // [rsp+D8h] [rbp-1148h]
  __int64 v75; // [rsp+E0h] [rbp-1140h]
  __int64 v76; // [rsp+E8h] [rbp-1138h]
  __int64 v77; // [rsp+F0h] [rbp-1130h]
  const __m128i *v78; // [rsp+F8h] [rbp-1128h] BYREF
  __m128i *v79; // [rsp+100h] [rbp-1120h]
  const __m128i *v80; // [rsp+108h] [rbp-1118h]
  __int64 v81; // [rsp+110h] [rbp-1110h]
  __int64 v82; // [rsp+118h] [rbp-1108h] BYREF
  int v83; // [rsp+120h] [rbp-1100h]
  __int64 v84; // [rsp+128h] [rbp-10F8h]
  _BYTE *v85; // [rsp+130h] [rbp-10F0h]
  __int64 v86; // [rsp+138h] [rbp-10E8h]
  _BYTE v87[1792]; // [rsp+140h] [rbp-10E0h] BYREF
  _BYTE *v88; // [rsp+840h] [rbp-9E0h]
  __int64 v89; // [rsp+848h] [rbp-9D8h]
  _BYTE v90[512]; // [rsp+850h] [rbp-9D0h] BYREF
  _BYTE *v91; // [rsp+A50h] [rbp-7D0h]
  __int64 v92; // [rsp+A58h] [rbp-7C8h]
  _BYTE v93[1792]; // [rsp+A60h] [rbp-7C0h] BYREF
  _BYTE *v94; // [rsp+1160h] [rbp-C0h]
  __int64 v95; // [rsp+1168h] [rbp-B8h]
  _BYTE v96[64]; // [rsp+1170h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+11B0h] [rbp-70h]
  __int64 v98; // [rsp+11B8h] [rbp-68h]
  int v99; // [rsp+11C0h] [rbp-60h]
  char v100; // [rsp+11E0h] [rbp-40h]

  v16 = a2;
  v64[0] = a5;
  v64[1] = a6;
  v55 = a15;
  v62 = a13 & 1;
  if ( a15 )
  {
    v61 = a16;
  }
  else
  {
    v61 = 0;
    v55 = a3 + 288;
  }
  v65 = 0;
  v66 = 0;
  v67 = 0;
  if ( a8 > 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::reserve");
  if ( a8 )
  {
    v65 = (const __m128i *)sub_22077B0(48 * a8);
    v66 = (__m128i *)v65;
    v67 = &v65[3 * a8];
  }
  v68 = 0u;
  v69 = 0u;
  v70 = 0u;
  if ( a8 )
  {
    v18 = 0;
    v19 = 0;
    do
    {
      while ( 1 )
      {
        v23 = 16 * v19;
        v63 = _mm_loadu_si128((const __m128i *)(a7 + 16 * v19));
        v24 = *(__int64 **)(a3 + 64);
        v68.m128i_i64[1] = *(_QWORD *)(a7 + 16 * v19);
        v69.m128i_i32[0] = v63.m128i_i32[2];
        v25 = (unsigned __int16 *)(*(_QWORD *)(v68.m128i_i64[1] + 48) + 16LL * v63.m128i_u32[2]);
        v26 = *v25;
        v27 = *((_QWORD *)v25 + 1);
        LOWORD(v71) = v26;
        v72 = v27;
        v69.m128i_i64[1] = sub_3007410((__int64)&v71, v24, v26, 16 * v19, a5, (__int64)a6);
        a6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a2 + 1128LL);
        v28 = a13 & 1;
        if ( a6 != sub_2FE3330 )
          v28 = a6((__int64)a2, v69.m128i_i64[1], v62);
        v70.m128i_i8[0] = v28 & 1 | v70.m128i_i8[0] & 0xFC | (2 * ((v28 ^ 1) & 1));
        if ( (a13 & 0x10) != 0 )
        {
          v21 = *(__int64 (**)())(*(_QWORD *)a2 + 1136LL);
          if ( v21 != sub_2FE3340
            && !((unsigned __int8 (__fastcall *)(_WORD *, _QWORD, _QWORD))v21)(
                  a2,
                  *(unsigned int *)(a9 + v23),
                  *(_QWORD *)(a9 + v23 + 8)) )
          {
            v70.m128i_i8[0] &= 0xFCu;
          }
        }
        v22 = v66;
        if ( v66 != v67 )
          break;
        sub_332CDC0((unsigned __int64 *)&v65, v66, &v68);
        v19 = ++v18;
        if ( v18 >= a8 )
          goto LABEL_17;
      }
      if ( v66 )
      {
        *v66 = _mm_loadu_si128(&v68);
        v22[1] = _mm_loadu_si128(&v69);
        v22[2] = _mm_loadu_si128(&v70);
        v22 = v66;
      }
      v66 = v22 + 3;
      v19 = ++v18;
    }
    while ( v18 < a8 );
LABEL_17:
    v16 = a2;
  }
  if ( a4 == 729 )
    sub_C64ED0("Unsupported library call operation!", 1u);
  v29 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v16 + 32LL);
  v30 = sub_2E79000(*(__int64 **)(a3 + 40));
  if ( v29 == sub_2D42F30 )
  {
    v31 = sub_AE2980(v30, 0)[1];
    v32 = 2;
    if ( v31 != 1 )
    {
      v32 = 3;
      if ( v31 != 2 )
      {
        v32 = 4;
        if ( v31 != 4 )
        {
          v32 = 5;
          if ( v31 != 8 )
          {
            v32 = 6;
            if ( v31 != 16 )
            {
              v32 = 7;
              if ( v31 != 32 )
              {
                v32 = 8;
                if ( v31 != 64 )
                  v32 = 9 * (v31 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v32 = v29((__int64)v16, v30, 0);
  }
  v56 = sub_33EED90(a3, *(const char **)&v16[4 * a4 + 262644], v32, 0);
  v54 = v33;
  v37 = sub_3007410((__int64)v64, *(__int64 **)(a3 + 64), v33, v34, v35, v36);
  v81 = a3;
  v60 = v37;
  v74 = 0xFFFFFFFF00000020LL;
  v85 = v87;
  v86 = 0x2000000000LL;
  v89 = 0x2000000000LL;
  v92 = 0x2000000000LL;
  v95 = 0x400000000LL;
  v38 = *(_QWORD *)v16;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v88 = v90;
  v91 = v93;
  v94 = v96;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v39 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v38 + 1128);
  if ( v39 == sub_2FE3330 )
  {
    v59 = !(a13 & 1);
    if ( (a13 & 0x10) == 0 )
      goto LABEL_30;
    v50 = *(__int64 (**)())(v38 + 1136);
    if ( v50 == sub_2FE3340 )
      goto LABEL_30;
    goto LABEL_54;
  }
  v62 = v39((__int64)v16, v60, v62);
  v59 = v62 ^ 1;
  if ( (a13 & 0x10) != 0 )
  {
    v50 = *(__int64 (**)())(*(_QWORD *)v16 + 1136LL);
    if ( v50 != sub_2FE3340 )
    {
LABEL_54:
      v51 = ((__int64 (__fastcall *)(_WORD *, _QWORD, __int64))v50)(v16, a11, a12);
      v52 = v59;
      if ( !v51 )
        v52 = 0;
      v59 = v52;
      v53 = v62;
      if ( !v51 )
        v53 = 0;
      v62 = v53;
    }
  }
  if ( v82 )
    sub_B91220((__int64)&v82, v82);
LABEL_30:
  v40 = *(_QWORD *)a14;
  v82 = v40;
  if ( v40 )
    sub_B96E90((__int64)&v82, v40, 1);
  v41 = *(_DWORD *)&v16[2 * a4 + 265564];
  v42 = v66;
  v66 = 0;
  v43 = *(_DWORD *)(a14 + 8);
  v44 = v65;
  LODWORD(v75) = v41;
  v83 = v43;
  v79 = v42;
  v71 = v55;
  v65 = 0;
  LODWORD(v72) = v61;
  v73 = v60;
  v76 = v56;
  v45 = (unsigned __int64)v78;
  v78 = v44;
  LODWORD(v77) = v54;
  HIDWORD(v74) = -1431655765 * (v42 - v44);
  v46 = v67;
  v67 = 0;
  v80 = v46;
  if ( v45 )
    j_j___libc_free_0(v45);
  v47 = *(void (****)())(v81 + 16);
  v48 = **v47;
  if ( v48 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v48)(v47, *(_QWORD *)(v81 + 40), v41, &v78);
  BYTE3(v74) = (a13 & 8) != 0;
  LOBYTE(v74) = v74 & 0xCC | (8 * a13) & 0x20 | (8 * a13) & 0x10 | (v62 | (2 * v59)) & 0x33;
  sub_3377410(a1, v16, (__int64)&v71);
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
  if ( v91 != v93 )
    _libc_free((unsigned __int64)v91);
  if ( v88 != v90 )
    _libc_free((unsigned __int64)v88);
  if ( v85 != v87 )
    _libc_free((unsigned __int64)v85);
  if ( v82 )
    sub_B91220((__int64)&v82, v82);
  if ( v78 )
    j_j___libc_free_0((unsigned __int64)v78);
  if ( v65 )
    j_j___libc_free_0((unsigned __int64)v65);
  return a1;
}
