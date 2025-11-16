// Function: sub_6D7930
// Address: 0x6d7930
//
__int64 __fastcall sub_6D7930(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rsi
  char *v7; // rdi
  __int64 v8; // rcx
  char v9; // al
  __int64 v10; // rax
  char v11; // dl
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __m128i v19; // xmm7
  __m128i v20; // xmm0
  __int8 v21; // al
  __int64 v22; // rax
  char *v23; // rdi
  _QWORD *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int8 v28; // r14
  __int32 v29; // ebx
  __int64 v30; // rax
  char *v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __m128i v36; // xmm2
  __m128i v37; // xmm3
  __m128i v38; // xmm4
  __m128i v39; // xmm5
  __m128i v40; // xmm6
  __m128i v41; // xmm7
  __m128i v42; // xmm1
  __m128i v43; // xmm2
  __m128i v44; // xmm3
  __m128i v45; // xmm4
  __m128i v46; // xmm5
  __m128i v47; // xmm6
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rax
  __m128i v51; // xmm1
  __m128i v52; // xmm2
  __m128i v53; // xmm3
  __m128i v54; // xmm7
  __m128i v55; // xmm4
  __m128i v56; // xmm5
  __m128i v57; // xmm6
  __int8 v58; // al
  __m128i v59; // xmm7
  _DWORD *v60; // rdi
  __m128i *v61; // rsi
  __int64 i; // rcx
  __int64 v63; // rax
  _DWORD *v64; // [rsp-8h] [rbp-258h]
  FILE v65; // [rsp+8h] [rbp-248h] BYREF
  __m128i v66; // [rsp+E0h] [rbp-170h] BYREF
  __m128i v67; // [rsp+F0h] [rbp-160h] BYREF
  __m128i v68; // [rsp+100h] [rbp-150h] BYREF
  __m128i v69; // [rsp+110h] [rbp-140h] BYREF
  __m128i v70; // [rsp+120h] [rbp-130h] BYREF
  __m128i v71; // [rsp+130h] [rbp-120h] BYREF
  __m128i v72; // [rsp+140h] [rbp-110h] BYREF
  __m128i v73; // [rsp+150h] [rbp-100h] BYREF
  __m128i v74; // [rsp+160h] [rbp-F0h] BYREF
  __m128i v75; // [rsp+170h] [rbp-E0h] BYREF
  __m128i v76; // [rsp+180h] [rbp-D0h] BYREF
  __m128i v77; // [rsp+190h] [rbp-C0h] BYREF
  __m128i v78; // [rsp+1A0h] [rbp-B0h] BYREF
  __m128i v79; // [rsp+1B0h] [rbp-A0h] BYREF
  __m128i v80; // [rsp+1C0h] [rbp-90h] BYREF
  __m128i v81; // [rsp+1D0h] [rbp-80h] BYREF
  __m128i v82; // [rsp+1E0h] [rbp-70h] BYREF
  __m128i v83; // [rsp+1F0h] [rbp-60h] BYREF
  __m128i v84; // [rsp+200h] [rbp-50h] BYREF
  __m128i v85; // [rsp+210h] [rbp-40h] BYREF
  __m128i v86[3]; // [rsp+220h] [rbp-30h] BYREF

  sub_6E1E00(4, &v65._IO_write_ptr, 0, 1);
  *(_BYTE *)(qword_4D03C50 + 20LL) |= 2u;
  *(_QWORD *)&v65._flags = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(4, &v65._IO_write_ptr, v2, v3);
  ++*(_BYTE *)(qword_4F061C8 + 34LL);
  sub_7B8B50(4, &v65._IO_write_ptr, v4, v5);
  sub_69ED20((__int64)&v65._unused2[4], 0, 0, 1);
  sub_7BE280(55, 53, 0, 0);
  v6 = 0;
  --*(_BYTE *)(qword_4F061C8 + 34LL);
  sub_6F69D0(&v65._unused2[4], 0);
  v7 = *(char **)&v65._unused2[4];
  if ( (unsigned int)sub_8D3A70(*(_QWORD *)&v65._unused2[4]) )
  {
    LODWORD(v65._IO_read_ptr) = 0;
    v22 = sub_72CD60(v7, 0);
    v7 = &v65._unused2[4];
    v6 = v22;
    sub_845C60(&v65._unused2[4], v22, 0, 2048, &v65._IO_read_ptr);
  }
  sub_6E2B30(v7, v6);
  if ( !v66.m128i_i8[0] )
    goto LABEL_8;
  v8 = *(_QWORD *)&v65._unused2[4];
  v9 = *(_BYTE *)(*(_QWORD *)&v65._unused2[4] + 140LL);
  if ( v9 == 12 )
  {
    v10 = *(_QWORD *)&v65._unused2[4];
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
      v11 = *(_BYTE *)(v10 + 140);
    }
    while ( v11 == 12 );
    if ( !v11 )
      goto LABEL_8;
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
      v9 = *(_BYTE *)(v8 + 140);
    }
    while ( v9 == 12 );
  }
  if ( !v9 )
  {
LABEL_8:
    sub_6E6260(a1);
    return sub_7BE280(26, 17, 0, 0);
  }
  if ( (unsigned int)sub_696840((__int64)&v65._unused2[4]) )
  {
    v13 = _mm_loadu_si128(&v66);
    v14 = _mm_loadu_si128(&v67);
    v15 = _mm_loadu_si128(&v68);
    v16 = _mm_loadu_si128(&v69);
    v17 = _mm_loadu_si128(&v70);
    *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)&v65._unused2[4]);
    v18 = _mm_loadu_si128(&v71);
    v19 = _mm_loadu_si128(&v72);
    *(__m128i *)(a1 + 16) = v13;
    v20 = _mm_loadu_si128(&v73);
    v21 = v66.m128i_i8[0];
    *(__m128i *)(a1 + 32) = v14;
    *(__m128i *)(a1 + 48) = v15;
    *(__m128i *)(a1 + 64) = v16;
    *(__m128i *)(a1 + 80) = v17;
    *(__m128i *)(a1 + 96) = v18;
    *(__m128i *)(a1 + 112) = v19;
    *(__m128i *)(a1 + 128) = v20;
    if ( v21 == 2 )
    {
      v36 = _mm_loadu_si128(&v75);
      v37 = _mm_loadu_si128(&v76);
      v38 = _mm_loadu_si128(&v77);
      v39 = _mm_loadu_si128(&v78);
      v40 = _mm_loadu_si128(&v79);
      *(__m128i *)(a1 + 144) = _mm_loadu_si128(&v74);
      v41 = _mm_loadu_si128(&v80);
      v42 = _mm_loadu_si128(&v81);
      *(__m128i *)(a1 + 160) = v36;
      *(__m128i *)(a1 + 176) = v37;
      v43 = _mm_loadu_si128(&v82);
      v44 = _mm_loadu_si128(&v83);
      *(__m128i *)(a1 + 192) = v38;
      v45 = _mm_loadu_si128(&v84);
      *(__m128i *)(a1 + 208) = v39;
      v46 = _mm_loadu_si128(&v85);
      *(__m128i *)(a1 + 224) = v40;
      v47 = _mm_loadu_si128(v86);
      *(__m128i *)(a1 + 240) = v41;
      *(__m128i *)(a1 + 256) = v42;
      *(__m128i *)(a1 + 272) = v43;
      *(__m128i *)(a1 + 288) = v44;
      *(__m128i *)(a1 + 304) = v45;
      *(__m128i *)(a1 + 320) = v46;
      *(__m128i *)(a1 + 336) = v47;
    }
    else if ( v21 == 5 || v21 == 1 )
    {
      *(_QWORD *)(a1 + 144) = v74.m128i_i64[0];
    }
    sub_6F4B70(a1);
    return sub_7BE280(26, 17, 0, 0);
  }
  v23 = *(char **)&v65._unused2[4];
  if ( !(unsigned int)sub_8D2630(*(_QWORD *)&v65._unused2[4], v6) )
  {
    sub_6E5E80(3358, (char *)v69.m128i_i64 + 4, *(_QWORD *)&v65._unused2[4]);
    goto LABEL_8;
  }
  if ( v66.m128i_i8[0] != 2 && (v6 = 1, v23 = &v65._unused2[4], !(unsigned int)sub_6F4D20(&v65._unused2[4], 1, 1))
    || v84.m128i_i8[13] != 15 )
  {
    sub_6E6000(v23, v6, v24, v25, v26, v27);
    sub_6E6260(a1);
    return sub_7BE280(26, 17, 0, 0);
  }
  v28 = v85.m128i_i8[0];
  if ( v85.m128i_i8[0] == 2 )
  {
    sub_6E6A50(v85.m128i_i64[1], a1);
    return sub_7BE280(26, 17, 0, 0);
  }
  v29 = v86[0].m128i_i32[0];
  if ( v86[0].m128i_i32[0] )
  {
    v24 = qword_4F04C68;
    v30 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    while ( v86[0].m128i_i32[0] != *(_DWORD *)v30 )
    {
      v30 -= 776;
      if ( !*(_BYTE *)(v30 + 780) )
      {
        if ( (unsigned int)sub_6E5430(v23, v6, qword_4F04C68, v25, v26) )
          sub_6851C0(0xD38u, &v69.m128i_i32[1]);
        goto LABEL_8;
      }
    }
  }
  if ( v85.m128i_i8[0] == 13 )
  {
    v48 = *(unsigned __int8 *)(v85.m128i_i64[1] + 24);
    if ( (_BYTE)v48 != 3 )
    {
      if ( (_BYTE)v48 != 20 )
      {
        if ( (_BYTE)v48 == 2 )
        {
          sub_6E6A50(*(_QWORD *)(v85.m128i_i64[1] + 56), a1);
          return sub_7BE280(26, 17, 0, 0);
        }
        if ( (unsigned int)sub_6E5430(v23, v6, v48, v25, v26) )
          sub_6851C0(0xD39u, &v69.m128i_i32[1]);
        goto LABEL_8;
      }
      v31 = *(char **)(v85.m128i_i64[1] + 56);
      goto LABEL_43;
    }
    v31 = *(char **)(v85.m128i_i64[1] + 56);
    goto LABEL_45;
  }
  if ( v85.m128i_i8[0] != 8 )
  {
    v31 = (char *)v85.m128i_i64[1];
    if ( v85.m128i_i8[0] != 7 )
    {
      if ( v85.m128i_i8[0] != 11 )
      {
        if ( (unsigned int)sub_6E5430(v23, v6, v24, v25, v26) )
        {
          v65._IO_read_end = v31;
          LOBYTE(v65._IO_read_ptr) = v28;
          LODWORD(v65._IO_read_base) = v29;
          v64 = sub_67D610(0xD28u, &v65, 8u);
          sub_67F190(
            (__int64)v64,
            (__int64)&v65,
            v32,
            v33,
            v34,
            v35,
            (char)v65._IO_read_ptr,
            (__int64)v65._IO_read_end,
            (int)v65._IO_read_base);
          sub_685910((__int64)v64, &v65);
        }
        goto LABEL_8;
      }
LABEL_43:
      v49 = sub_6E50B0(*(_QWORD *)v31, (char *)v69.m128i_i64 + 4);
      sub_6EAB60(*(_QWORD *)v31, 0, 1, (unsigned int)&v69.m128i_u32[1], (unsigned int)&dword_4F061D8, v49, a1);
      return sub_7BE280(26, 17, 0, 0);
    }
LABEL_45:
    if ( dword_4F04C44 == -1 )
    {
      v63 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v63 + 6) & 6) == 0
        && *(_BYTE *)(v63 + 4) != 12
        && (v31[170] & 0x10) != 0
        && (unsigned int)sub_8D23E0(*((_QWORD *)v31 + 15)) )
      {
        sub_8AC4A0(v31, v6);
      }
    }
    v50 = sub_6E50B0(*(_QWORD *)v31, (char *)v69.m128i_i64 + 4);
    sub_6F8E70(v31, (char *)v69.m128i_i64 + 4, &dword_4F061D8, a1, v50);
    return sub_7BE280(26, 17, 0, 0);
  }
  v51 = _mm_loadu_si128(&v67);
  v52 = _mm_loadu_si128(&v68);
  v53 = _mm_loadu_si128(&v69);
  *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)&v65._unused2[4]);
  v54 = _mm_loadu_si128(&v66);
  v55 = _mm_loadu_si128(&v70);
  v56 = _mm_loadu_si128(&v71);
  v57 = _mm_loadu_si128(&v72);
  *(__m128i *)(a1 + 32) = v51;
  *(__m128i *)(a1 + 16) = v54;
  v58 = v66.m128i_i8[0];
  v59 = _mm_loadu_si128(&v73);
  *(__m128i *)(a1 + 48) = v52;
  *(__m128i *)(a1 + 64) = v53;
  *(__m128i *)(a1 + 80) = v55;
  *(__m128i *)(a1 + 96) = v56;
  *(__m128i *)(a1 + 112) = v57;
  *(__m128i *)(a1 + 128) = v59;
  if ( v58 == 2 )
  {
    v60 = (_DWORD *)(a1 + 144);
    v61 = &v74;
    for ( i = 52; i; --i )
    {
      *v60 = v61->m128i_i32[0];
      v61 = (__m128i *)((char *)v61 + 4);
      ++v60;
    }
  }
  else if ( v58 == 5 || v58 == 1 )
  {
    *(_QWORD *)(a1 + 144) = v74.m128i_i64[0];
  }
  return sub_7BE280(26, 17, 0, 0);
}
