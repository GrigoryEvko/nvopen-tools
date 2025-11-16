// Function: sub_3031C30
// Address: 0x3031c30
//
void __fastcall sub_3031C30(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  unsigned __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v8; // rsi
  __int16 *v9; // rdx
  __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // r15
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rbx
  int v16; // r14d
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r11
  __int64 *v20; // rax
  unsigned __int64 v21; // rsi
  int v22; // edx
  int v23; // ecx
  __int64 v24; // rax
  __m128i v25; // xmm2
  __m128i v26; // xmm1
  __m128i v27; // xmm0
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __int64 v30; // r11
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // r8
  __int128 v34; // rax
  unsigned __int64 v35; // rcx
  unsigned int v36; // r12d
  unsigned __int64 v37; // r13
  __int64 v38; // r14
  unsigned __int64 v39; // rbx
  __int64 *v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // r10
  __int64 v44; // rax
  __int64 v45; // r11
  __int64 v46; // r9
  __int64 *v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 *v50; // rdx
  _OWORD *v51; // rdi
  __m128i v52; // xmm5
  __int64 v53; // [rsp+10h] [rbp-1C0h]
  __int64 v55; // [rsp+20h] [rbp-1B0h]
  __int64 v56; // [rsp+20h] [rbp-1B0h]
  __int64 v57; // [rsp+20h] [rbp-1B0h]
  __int64 v58; // [rsp+28h] [rbp-1A8h]
  unsigned __int64 v59; // [rsp+30h] [rbp-1A0h]
  __int64 v60; // [rsp+30h] [rbp-1A0h]
  __int64 v62; // [rsp+38h] [rbp-198h]
  __int64 v63; // [rsp+38h] [rbp-198h]
  __int64 v64; // [rsp+40h] [rbp-190h] BYREF
  int v65; // [rsp+48h] [rbp-188h]
  int v66; // [rsp+50h] [rbp-180h] BYREF
  __int64 v67; // [rsp+58h] [rbp-178h]
  __m128i v68; // [rsp+60h] [rbp-170h] BYREF
  _OWORD v69[4]; // [rsp+70h] [rbp-160h] BYREF
  _BYTE *v70; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-118h]
  _BYTE v72[80]; // [rsp+C0h] [rbp-110h] BYREF
  _OWORD *v73; // [rsp+110h] [rbp-C0h]
  __int64 v74; // [rsp+118h] [rbp-B8h]
  _OWORD v75[3]; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v76; // [rsp+150h] [rbp-80h]
  __m128i v77; // [rsp+160h] [rbp-70h]

  v8 = *(_QWORD *)(a1 + 80);
  v64 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v64, v8, 1);
  v9 = *(__int16 **)(a1 + 48);
  v65 = *(_DWORD *)(a1 + 72);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v66) = v10;
  v67 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
      goto LABEL_5;
    if ( (unsigned __int16)(v10 - 176) > 0x34u )
      goto LABEL_39;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v66) )
      goto LABEL_5;
    if ( !sub_3007100((__int64)&v66) )
      goto LABEL_10;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v66 )
  {
    if ( (unsigned __int16)(v66 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_39:
    v12 = word_4456340[(unsigned __int16)v66 - 1];
    goto LABEL_11;
  }
LABEL_10:
  v12 = (unsigned int)sub_3007130((__int64)&v66, v8);
LABEL_11:
  v70 = v72;
  v71 = 0x500000000LL;
  if ( (_DWORD)v12 )
  {
    v59 = v5;
    v13 = 5;
    v14 = 0;
    v15 = v6;
    v16 = 0;
    while ( 1 )
    {
      LOWORD(v15) = 7;
      if ( v14 + 1 > v13 )
      {
        sub_C8D5F0((__int64)&v70, v72, v14 + 1, 0x10u, a5, v14 + 1);
        v14 = (unsigned int)v71;
      }
      v17 = &v70[16 * v14];
      ++v16;
      *v17 = v15;
      v17[1] = 0;
      v14 = (unsigned int)(v71 + 1);
      LODWORD(v71) = v71 + 1;
      if ( v16 == (_DWORD)v12 )
        break;
      v13 = HIDWORD(v71);
    }
    v5 = v59;
    v18 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL);
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 24LL);
    if ( HIDWORD(v71) < (unsigned __int64)(v14 + 1) )
    {
      v53 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL);
      v60 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 24LL);
      sub_C8D5F0((__int64)&v70, v72, v14 + 1, 0x10u, v18, v14 + 1);
      v14 = (unsigned int)v71;
      v18 = v53;
      v19 = v60;
    }
  }
  else
  {
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 24LL);
    v18 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL);
    v14 = 0;
  }
  v20 = (__int64 *)&v70[16 * v14];
  *v20 = v18;
  v21 = (unsigned __int64)v70;
  v20[1] = v19;
  LODWORD(v71) = v71 + 1;
  v23 = sub_33E5830(a2, v21);
  v24 = *(_QWORD *)(a1 + 40);
  v25 = _mm_loadu_si128((const __m128i *)v24);
  v73 = v75;
  v74 = 0x800000003LL;
  v68 = v25;
  v26 = _mm_loadu_si128((const __m128i *)(v24 + 40));
  v75[0] = v25;
  v69[0] = v26;
  v27 = _mm_loadu_si128((const __m128i *)(v24 + 80));
  v75[1] = v26;
  v69[1] = v27;
  v75[2] = v27;
  if ( a4 )
  {
    v28 = _mm_loadu_si128((const __m128i *)(v24 + 120));
    v29 = _mm_loadu_si128((const __m128i *)(v24 + 160));
    LODWORD(v74) = 5;
    v30 = 5;
    v76 = v28;
    v77 = v29;
  }
  else
  {
    v52 = _mm_loadu_si128((const __m128i *)(v24 + 120));
    v30 = 4;
    LODWORD(v74) = 4;
    v76 = v52;
  }
  v31 = sub_33EA9D0(
          a2,
          47,
          (unsigned int)&v64,
          v23,
          v22,
          *(_QWORD *)(a1 + 112),
          (__int64)v75,
          v30,
          *(unsigned __int16 *)(a1 + 96),
          *(_QWORD *)(a1 + 104));
  v68.m128i_i64[0] = (__int64)v69;
  v32 = v31;
  v68.m128i_i64[1] = 0x400000000LL;
  if ( (_DWORD)v12 )
  {
    v33 = 0;
    v55 = a3;
    *((_QWORD *)&v34 + 1) = 0;
    v35 = 4;
    v36 = 0;
    v37 = v5;
    v38 = v31;
    while ( 1 )
    {
      v39 = v37 & 0xFFFFFFFF00000000LL | v36;
      v37 = v39;
      if ( *((_QWORD *)&v34 + 1) + 1LL > v35 )
      {
        sub_C8D5F0((__int64)&v68, v69, *((_QWORD *)&v34 + 1) + 1LL, 0x10u, v33, *((_QWORD *)&v34 + 1) + 1LL);
        *((_QWORD *)&v34 + 1) = v68.m128i_u32[2];
      }
      v40 = (__int64 *)(v68.m128i_i64[0] + 16LL * *((_QWORD *)&v34 + 1));
      ++v36;
      *v40 = v38;
      v40[1] = v39;
      *((_QWORD *)&v34 + 1) = (unsigned int)++v68.m128i_i32[2];
      if ( v36 == (_DWORD)v12 )
        break;
      v35 = v68.m128i_u32[3];
    }
    v32 = v38;
    a3 = v55;
    *(_QWORD *)&v34 = v68.m128i_i64[0];
  }
  else
  {
    *((_QWORD *)&v34 + 1) = 0;
    *(_QWORD *)&v34 = v69;
  }
  v56 = v32;
  v43 = sub_33FC220(a2, 156, (unsigned int)&v64, v66, v67, v32, v34);
  v44 = *(unsigned int *)(a3 + 8);
  v45 = v41;
  v46 = v56;
  if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v57 = v43;
    v58 = v41;
    v62 = v46;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v44 + 1, 0x10u, v42, v46);
    v44 = *(unsigned int *)(a3 + 8);
    v43 = v57;
    v45 = v58;
    v46 = v62;
  }
  v47 = (__int64 *)(*(_QWORD *)a3 + 16 * v44);
  *v47 = v43;
  v47[1] = v45;
  v48 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  v49 = *(unsigned int *)(a3 + 12);
  *(_DWORD *)(a3 + 8) = v48;
  if ( v48 + 1 > v49 )
  {
    v63 = v46;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v48 + 1, 0x10u, v48 + 1, v46);
    v48 = *(unsigned int *)(a3 + 8);
    v46 = v63;
  }
  v50 = (__int64 *)(*(_QWORD *)a3 + 16 * v48);
  *v50 = v46;
  v50[1] = v12;
  v51 = (_OWORD *)v68.m128i_i64[0];
  ++*(_DWORD *)(a3 + 8);
  if ( v51 != v69 )
    _libc_free((unsigned __int64)v51);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
LABEL_5:
  if ( v64 )
    sub_B91220((__int64)&v64, v64);
}
