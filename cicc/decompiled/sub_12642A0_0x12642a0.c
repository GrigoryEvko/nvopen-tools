// Function: sub_12642A0
// Address: 0x12642a0
//
__int64 __fastcall sub_12642A0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 (*v5)(void); // rax
  unsigned int v6; // r14d
  char v7; // al
  __int64 v8; // rax
  __m128i si128; // xmm0
  int v10; // eax
  __int64 v11; // rax
  __m128i v12; // xmm0
  bool v13; // zf
  __int64 v14; // rax
  __m128i v15; // xmm0
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // r14
  int v20; // eax
  __int64 v21; // rdi
  __int64 i; // rdx
  __int64 j; // r15
  __int64 v24; // r12
  __int64 v25; // rax
  __m128i *v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rdi
  __m128i *v30; // rbx
  __m128i *v31; // r13
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rax
  __m128i v37; // xmm0
  int v38; // [rsp+24h] [rbp-27Ch]
  __int64 v40; // [rsp+48h] [rbp-258h]
  __int64 *v41; // [rsp+48h] [rbp-258h]
  __int64 v42; // [rsp+50h] [rbp-250h] BYREF
  __int64 v43; // [rsp+58h] [rbp-248h] BYREF
  __m128i *v44; // [rsp+60h] [rbp-240h] BYREF
  __m128i *v45; // [rsp+68h] [rbp-238h]
  __int64 v46; // [rsp+70h] [rbp-230h]
  __m128i v47; // [rsp+80h] [rbp-220h] BYREF
  _QWORD v48[2]; // [rsp+90h] [rbp-210h] BYREF
  _BYTE v49[80]; // [rsp+A0h] [rbp-200h] BYREF
  __m128i v50; // [rsp+F0h] [rbp-1B0h] BYREF
  _QWORD v51[2]; // [rsp+100h] [rbp-1A0h] BYREF
  unsigned __int64 v52; // [rsp+110h] [rbp-190h]
  __int64 v53; // [rsp+118h] [rbp-188h]
  unsigned __int64 v54; // [rsp+120h] [rbp-180h]
  __int64 v55; // [rsp+128h] [rbp-178h]
  _BYTE v56[8]; // [rsp+130h] [rbp-170h] BYREF
  int v57; // [rsp+138h] [rbp-168h]
  _QWORD v58[2]; // [rsp+140h] [rbp-160h] BYREF
  _QWORD v59[2]; // [rsp+150h] [rbp-150h] BYREF
  _QWORD v60[28]; // [rsp+160h] [rbp-140h] BYREF
  __int16 v61; // [rsp+240h] [rbp-60h]
  __int64 v62; // [rsp+248h] [rbp-58h]
  __int64 v63; // [rsp+250h] [rbp-50h]
  __int64 v64; // [rsp+258h] [rbp-48h]
  __int64 v65; // [rsp+260h] [rbp-40h]

  sub_1C13840(v49, sub_12631C0, sub_12631B0, 0, 0);
  v4 = *a2;
  *a2 = 0;
  v50.m128i_i64[0] = v4;
  sub_1C26C10(&v42, &v50, v49);
  if ( v50.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v50.m128i_i64[0] + 8LL))(v50.m128i_i64[0]);
  if ( v42 )
  {
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v50.m128i_i64[0] = (__int64)v51;
    strcpy((char *)v51, "-march=nvptx");
    v50.m128i_i64[1] = 12;
    sub_8F9C20(&v44, &v50);
    if ( (_QWORD *)v50.m128i_i64[0] != v51 )
      j_j___libc_free_0(v50.m128i_i64[0], v51[0] + 1LL);
    v5 = *(__int64 (**)(void))(*(_QWORD *)v42 + 8LL);
    if ( (char *)v5 == (char *)sub_12631A0 )
      v40 = *(_QWORD *)(v42 + 32);
    else
      v40 = v5();
    v6 = *(_DWORD *)v40 / 10;
    sub_222DF20(v60);
    v60[27] = 0;
    v61 = 0;
    v62 = 0;
    v60[0] = off_4A06798;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v50.m128i_i64[0] = (__int64)qword_4A071C8;
    *(__int64 *)((char *)v50.m128i_i64 + qword_4A071C8[-3]) = (__int64)&unk_4A071F0;
    sub_222DD70(&v50.m128i_i8[*(_QWORD *)(v50.m128i_i64[0] - 24)], 0);
    v51[0] = 0;
    v51[1] = 0;
    v52 = 0;
    v50.m128i_i64[0] = (__int64)off_4A07238;
    v53 = 0;
    v54 = 0;
    v60[0] = off_4A07260;
    v55 = 0;
    v50.m128i_i64[1] = (__int64)off_4A07480;
    sub_220A990(v56);
    v57 = 16;
    LOBYTE(v59[0]) = 0;
    v50.m128i_i64[1] = (__int64)off_4A07080;
    v58[0] = v59;
    v58[1] = 0;
    sub_222DD70(v60, &v50.m128i_u64[1]);
    sub_223E0D0(&v50, "-mcpu=sm_", 9);
    sub_223E730(&v50, v6);
    v47.m128i_i64[0] = (__int64)v48;
    v47.m128i_i64[1] = 0;
    LOBYTE(v48[0]) = 0;
    if ( v54 )
    {
      if ( v54 > v52 )
        sub_2241130(&v47, 0, 0, v53, v54 - v53);
      else
        sub_2241130(&v47, 0, 0, v53, v52 - v53);
    }
    else
    {
      sub_2240AE0(&v47, v58);
    }
    sub_8F9C20(&v44, &v47);
    if ( (_QWORD *)v47.m128i_i64[0] != v48 )
      j_j___libc_free_0(v47.m128i_i64[0], v48[0] + 1LL);
    v7 = *(_BYTE *)(v40 + 200);
    if ( (v7 & 0x20) != 0 )
    {
      v47.m128i_i64[0] = (__int64)v48;
      strcpy((char *)v48, "-nvptx-f32ftz");
      v47.m128i_i64[1] = 13;
      sub_8F9C20(&v44, &v47);
      if ( (_QWORD *)v47.m128i_i64[0] != v48 )
        j_j___libc_free_0(v47.m128i_i64[0], v48[0] + 1LL);
      v7 = *(_BYTE *)(v40 + 200);
    }
    v47.m128i_i64[0] = (__int64)v48;
    v43 = 18;
    if ( v7 < 0 )
    {
      v8 = sub_22409D0(&v47, &v43, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F0F550);
      v47.m128i_i64[0] = v8;
      v48[0] = v43;
      *(_WORD *)(v8 + 16) = 12605;
    }
    else
    {
      v8 = sub_22409D0(&v47, &v43, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F0F550);
      v47.m128i_i64[0] = v8;
      v48[0] = v43;
      *(_WORD *)(v8 + 16) = 12349;
    }
    *(__m128i *)v8 = si128;
    v47.m128i_i64[1] = v43;
    *(_BYTE *)(v47.m128i_i64[0] + v43) = 0;
    sub_8F9C20(&v44, &v47);
    if ( (_QWORD *)v47.m128i_i64[0] != v48 )
      j_j___libc_free_0(v47.m128i_i64[0], v48[0] + 1LL);
    v10 = *(_DWORD *)(v40 + 204);
    if ( v10 == 1 )
    {
      v47.m128i_i64[0] = (__int64)v48;
      v43 = 20;
      v36 = sub_22409D0(&v47, &v43, 0);
      v37 = _mm_load_si128((const __m128i *)&xmmword_3F0F560);
      v47.m128i_i64[0] = v36;
      v48[0] = v43;
      *(_DWORD *)(v36 + 16) = 842871347;
      *(__m128i *)v36 = v37;
    }
    else
    {
      if ( v10 != 2 )
        goto LABEL_23;
      v47.m128i_i64[0] = (__int64)v48;
      v43 = 20;
      v11 = sub_22409D0(&v47, &v43, 0);
      v12 = _mm_load_si128((const __m128i *)&xmmword_3F0F560);
      v47.m128i_i64[0] = v11;
      v48[0] = v43;
      *(_DWORD *)(v11 + 16) = 826094131;
      *(__m128i *)v11 = v12;
    }
    v47.m128i_i64[1] = v43;
    *(_BYTE *)(v47.m128i_i64[0] + v43) = 0;
    sub_8F9C20(&v44, &v47);
    if ( (_QWORD *)v47.m128i_i64[0] != v48 )
      j_j___libc_free_0(v47.m128i_i64[0], v48[0] + 1LL);
LABEL_23:
    v13 = (*(_BYTE *)(v40 + 200) & 0x40) == 0;
    v47.m128i_i64[0] = (__int64)v48;
    v43 = 21;
    if ( v13 )
    {
      v14 = sub_22409D0(&v47, &v43, 0);
      v15 = _mm_load_si128((const __m128i *)&xmmword_3F0F570);
      v47.m128i_i64[0] = v14;
      v48[0] = v43;
      *(_DWORD *)(v14 + 16) = 1026700134;
      *(_BYTE *)(v14 + 20) = 49;
    }
    else
    {
      v14 = sub_22409D0(&v47, &v43, 0);
      v15 = _mm_load_si128((const __m128i *)&xmmword_3F0F570);
      v47.m128i_i64[0] = v14;
      v48[0] = v43;
      *(_DWORD *)(v14 + 16) = 1026700134;
      *(_BYTE *)(v14 + 20) = 48;
    }
    *(__m128i *)v14 = v15;
    v16 = (__int64)&v47;
    v47.m128i_i64[1] = v43;
    *(_BYTE *)(v47.m128i_i64[0] + v43) = 0;
    sub_8F9C20(&v44, &v47);
    if ( (_QWORD *)v47.m128i_i64[0] != v48 )
    {
      v16 = v48[0] + 1LL;
      j_j___libc_free_0(v47.m128i_i64[0], v48[0] + 1LL);
    }
    if ( *(_DWORD *)(v40 + 4) == 2 )
    {
      v47.m128i_i64[0] = (__int64)v48;
      v16 = (__int64)&v47;
      strcpy((char *)v48, "--device-c");
      v47.m128i_i64[1] = 10;
      sub_8F9C20(&v44, &v47);
      if ( (_QWORD *)v47.m128i_i64[0] != v48 )
      {
        v16 = v48[0] + 1LL;
        j_j___libc_free_0(v47.m128i_i64[0], v48[0] + 1LL);
      }
    }
    v17 = ((char *)v45 - (char *)v44) >> 5;
    v38 = v17 + *(_DWORD *)(a3 + 68);
    v18 = 8LL * v38;
    if ( (unsigned __int64)v38 > 0xFFFFFFFFFFFFFFFLL )
      v18 = -1;
    v19 = sub_2207820(v18);
    v20 = *(_DWORD *)(a3 + 68);
    if ( v20 <= 0 )
    {
      if ( (int)v17 <= 0 )
        goto LABEL_37;
    }
    else
    {
      v21 = *(_QWORD *)(a3 + 72);
      v16 = v20;
      for ( i = 0; i != v20; ++i )
        *(_QWORD *)(v19 + 8 * i) = *(_QWORD *)(v21 + 8 * i);
      if ( (int)v17 <= 0 )
        goto LABEL_38;
    }
    for ( j = 0; ; ++j )
    {
      v24 = v44[2 * j].m128i_i64[1];
      v41 = (__int64 *)(v19 + 8LL * ((int)j + v20));
      v25 = sub_2207820(v24 + 1);
      v26 = v44;
      *v41 = v25;
      v16 = *(_QWORD *)(v19 + 8LL * ((int)j + *(_DWORD *)(a3 + 68)));
      sub_2241570(&v26[2 * j], v16, v24, 0);
      *(_BYTE *)(*(_QWORD *)(v19 + 8LL * ((int)j + *(_DWORD *)(a3 + 68))) + v24) = 0;
      if ( (_DWORD)v17 - 1 == j )
        break;
      v20 = *(_DWORD *)(a3 + 68);
    }
LABEL_37:
    v21 = *(_QWORD *)(a3 + 72);
    if ( !v21 )
    {
LABEL_39:
      *(_QWORD *)(a3 + 72) = v19;
      *(_DWORD *)(a3 + 68) = v38;
      v27 = v42;
      *(_WORD *)(v42 + 136) = 0;
      v28 = *(_QWORD *)(v27 + 16);
      LOBYTE(v27) = *(_BYTE *)(a1 + 8);
      v29 = (_QWORD *)v58[0];
      *(_QWORD *)a1 = v28;
      *(_BYTE *)(a1 + 8) = v27 & 0xFC | 2;
      v50.m128i_i64[0] = (__int64)off_4A07238;
      v60[0] = off_4A07260;
      v50.m128i_i64[1] = (__int64)off_4A07080;
      if ( v29 != v59 )
      {
        v16 = v59[0] + 1LL;
        j_j___libc_free_0(v29, v59[0] + 1LL);
      }
      v50.m128i_i64[1] = (__int64)off_4A07480;
      sub_2209150(v56, v16, v28);
      v50.m128i_i64[0] = (__int64)qword_4A071C8;
      *(__int64 *)((char *)v50.m128i_i64 + qword_4A071C8[-3]) = (__int64)&unk_4A071F0;
      v60[0] = off_4A06798;
      sub_222E050(v60);
      v30 = v45;
      v31 = v44;
      if ( v45 != v44 )
      {
        do
        {
          if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
            j_j___libc_free_0(v31->m128i_i64[0], v31[1].m128i_i64[0] + 1);
          v31 += 2;
        }
        while ( v30 != v31 );
        v31 = v44;
      }
      if ( v31 )
        j_j___libc_free_0(v31, v46 - (_QWORD)v31);
      goto LABEL_48;
    }
LABEL_38:
    j_j___libc_free_0_0(v21);
    goto LABEL_39;
  }
  v33 = sub_14EE0B0();
  LOWORD(v51[0]) = 259;
  v50.m128i_i64[0] = (__int64)"Invalid NVVM IR Container";
  v34 = sub_22077B0(56);
  v35 = v34;
  if ( v34 )
    sub_16BCC70(v34, &v50, 1, v33);
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v35 & 0xFFFFFFFFFFFFFFFELL;
LABEL_48:
  if ( v42 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 56LL))(v42);
  return a1;
}
