// Function: sub_2940540
// Address: 0x2940540
//
__int64 __fastcall sub_2940540(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rbx
  __int32 v7; // r15d
  _BYTE *v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  unsigned int v11; // ebx
  __int64 v12; // r13
  __int64 v13; // rax
  __m128i v14; // rax
  char v15; // al
  __m128i *v16; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // ebx
  __int64 v21; // r13
  __int64 v22; // r14
  __int64 v23; // rax
  __m128i v24; // rax
  char v25; // al
  __m128i *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __m128i v29; // xmm2
  __m128i v30; // xmm6
  __m128i v31; // xmm4
  __int64 v32; // [rsp+38h] [rbp-448h]
  __int64 v33; // [rsp+40h] [rbp-440h]
  __int64 v34; // [rsp+40h] [rbp-440h]
  unsigned __int8 v35; // [rsp+4Fh] [rbp-431h]
  __int64 v36; // [rsp+58h] [rbp-428h]
  __int64 v37; // [rsp+58h] [rbp-428h]
  __m128i v38; // [rsp+70h] [rbp-410h] BYREF
  __m128i v39; // [rsp+80h] [rbp-400h] BYREF
  __int64 v40; // [rsp+90h] [rbp-3F0h]
  __m128i v41[2]; // [rsp+A0h] [rbp-3E0h] BYREF
  unsigned __int8 v42; // [rsp+C0h] [rbp-3C0h]
  __m128i v43; // [rsp+D0h] [rbp-3B0h] BYREF
  __m128i v44; // [rsp+E0h] [rbp-3A0h]
  __int64 v45; // [rsp+F0h] [rbp-390h]
  __m128i v46; // [rsp+100h] [rbp-380h] BYREF
  __m128i v47; // [rsp+110h] [rbp-370h] BYREF
  __int64 v48; // [rsp+120h] [rbp-360h]
  __m128i v49; // [rsp+130h] [rbp-350h] BYREF
  __m128i v50; // [rsp+140h] [rbp-340h] BYREF
  __int64 v51; // [rsp+150h] [rbp-330h]
  __m128i v52; // [rsp+160h] [rbp-320h] BYREF
  __m128i v53; // [rsp+170h] [rbp-310h]
  __int64 v54; // [rsp+180h] [rbp-300h]
  _BYTE *v55; // [rsp+190h] [rbp-2F0h] BYREF
  __int64 v56; // [rsp+198h] [rbp-2E8h]
  _BYTE v57[64]; // [rsp+1A0h] [rbp-2E0h] BYREF
  unsigned int *v58[2]; // [rsp+1E0h] [rbp-2A0h] BYREF
  char v59; // [rsp+1F0h] [rbp-290h] BYREF
  void *v60; // [rsp+260h] [rbp-220h]
  __m128i v61[5]; // [rsp+270h] [rbp-210h] BYREF
  char *v62; // [rsp+2C0h] [rbp-1C0h]
  char v63; // [rsp+2D0h] [rbp-1B0h] BYREF
  __m128i v64[5]; // [rsp+310h] [rbp-170h] BYREF
  char *v65; // [rsp+360h] [rbp-120h]
  char v66; // [rsp+370h] [rbp-110h] BYREF
  __m128i v67; // [rsp+3B0h] [rbp-D0h] BYREF
  __m128i v68; // [rsp+3C0h] [rbp-C0h]
  __int64 v69; // [rsp+3D0h] [rbp-B0h]
  char *v70; // [rsp+400h] [rbp-80h]
  char v71; // [rsp+410h] [rbp-70h] BYREF

  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, (unsigned __int8 *)a2) )
    return 0;
  sub_2939E80((__int64)v41, a1, *(_QWORD *)(a2 + 8));
  v35 = v42;
  if ( !v42 )
    return 0;
  v2 = *(_QWORD *)(a2 - 96);
  v45 = 0;
  v43 = 0;
  v44 = 0;
  v3 = *(_QWORD *)(v2 + 8);
  if ( *(_BYTE *)(v3 + 8) == 17 )
  {
    sub_2939E80((__int64)&v38, a1, v3);
    v29 = _mm_loadu_si128(&v39);
    v43 = _mm_loadu_si128(&v38);
    v45 = v40;
    v44 = v29;
    if ( !(_BYTE)v40 || v43.m128i_i32[2] != v41[0].m128i_i32[2] )
      return 0;
  }
  sub_23D0AB0((__int64)v58, a2, 0, 0, 0);
  sub_293CE40(v61, (_QWORD *)a1, a2, *(_QWORD *)(a2 - 64), v41);
  sub_293CE40(v64, (_QWORD *)a1, a2, *(_QWORD *)(a2 - 32), v41);
  v6 = v41[0].m128i_u32[3];
  v7 = v41[0].m128i_i32[3];
  v55 = v57;
  v56 = 0x800000000LL;
  if ( v41[0].m128i_i32[3] )
  {
    v8 = v57;
    v9 = v57;
    if ( v41[0].m128i_u32[3] > 8uLL )
    {
      sub_C8D5F0((__int64)&v55, v57, v41[0].m128i_u32[3], 8u, v4, v5);
      v8 = v55;
      v9 = &v55[8 * (unsigned int)v56];
    }
    for ( i = &v8[8 * v6]; i != v9; ++v9 )
    {
      if ( v9 )
        *v9 = 0;
    }
    LODWORD(v56) = v7;
    if ( !(_BYTE)v45 )
    {
      if ( v41[0].m128i_i32[3] )
      {
        v11 = 0;
        v34 = *(_QWORD *)(a2 - 96);
        do
        {
          v12 = sub_293BC00((__int64)v61, v11);
          v13 = sub_293BC00((__int64)v64, v11);
          v52.m128i_i32[0] = v11;
          v36 = v13;
          LOWORD(v54) = 265;
          v14.m128i_i64[0] = (__int64)sub_BD5D20(a2);
          v49 = v14;
          LOWORD(v51) = 773;
          v50.m128i_i64[0] = (__int64)".i";
          v15 = v54;
          if ( (_BYTE)v54 )
          {
            if ( (_BYTE)v54 == 1 )
            {
              v30 = _mm_loadu_si128(&v50);
              v67 = _mm_loadu_si128(&v49);
              v69 = v51;
              v68 = v30;
            }
            else
            {
              if ( BYTE1(v54) == 1 )
              {
                v32 = v52.m128i_i64[1];
                v16 = (__m128i *)v52.m128i_i64[0];
              }
              else
              {
                v16 = &v52;
                v15 = 2;
              }
              v68.m128i_i64[0] = (__int64)v16;
              LOBYTE(v69) = 2;
              v67.m128i_i64[0] = (__int64)&v49;
              BYTE1(v69) = v15;
              v68.m128i_i64[1] = v32;
            }
          }
          else
          {
            LOWORD(v69) = 256;
          }
          v18 = sub_B36550(v58, v34, v12, v36, (__int64)&v67, 0);
          v19 = v11++;
          *(_QWORD *)&v55[8 * v19] = v18;
        }
        while ( v41[0].m128i_i32[3] > v11 );
      }
      goto LABEL_24;
    }
LABEL_33:
    v20 = 0;
    sub_293CE40(&v67, (_QWORD *)a1, a2, *(_QWORD *)(a2 - 96), &v43);
    if ( v41[0].m128i_i32[3] )
    {
      do
      {
        v21 = sub_293BC00((__int64)&v67, v20);
        v22 = sub_293BC00((__int64)v61, v20);
        v23 = sub_293BC00((__int64)v64, v20);
        v49.m128i_i32[0] = v20;
        v37 = v23;
        LOWORD(v51) = 265;
        v24.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v46 = v24;
        LOWORD(v48) = 773;
        v47.m128i_i64[0] = (__int64)".i";
        v25 = v51;
        if ( (_BYTE)v51 )
        {
          if ( (_BYTE)v51 == 1 )
          {
            v31 = _mm_loadu_si128(&v47);
            v52 = _mm_loadu_si128(&v46);
            v54 = v48;
            v53 = v31;
          }
          else
          {
            if ( BYTE1(v51) == 1 )
            {
              v33 = v49.m128i_i64[1];
              v26 = (__m128i *)v49.m128i_i64[0];
            }
            else
            {
              v26 = &v49;
              v25 = 2;
            }
            v53.m128i_i64[0] = (__int64)v26;
            LOBYTE(v54) = 2;
            v52.m128i_i64[0] = (__int64)&v46;
            BYTE1(v54) = v25;
            v53.m128i_i64[1] = v33;
          }
        }
        else
        {
          LOWORD(v54) = 256;
        }
        v27 = sub_B36550(v58, v21, v22, v37, (__int64)&v52, 0);
        v28 = v20++;
        *(_QWORD *)&v55[8 * v28] = v27;
      }
      while ( v41[0].m128i_i32[3] > v20 );
    }
    if ( v70 != &v71 )
      _libc_free((unsigned __int64)v70);
    goto LABEL_24;
  }
  if ( (_BYTE)v45 )
    goto LABEL_33;
LABEL_24:
  sub_293CAB0(a1, a2, (__int64)&v55, (__int64)v41);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( v65 != &v66 )
    _libc_free((unsigned __int64)v65);
  if ( v62 != &v63 )
    _libc_free((unsigned __int64)v62);
  nullsub_61();
  v60 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v58[0] != &v59 )
    _libc_free((unsigned __int64)v58[0]);
  return v35;
}
