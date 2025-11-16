// Function: sub_36E4A00
// Address: 0x36e4a00
//
void __fastcall sub_36E4A00(__int64 a1, __int64 a2, char a3, __m128i a4)
{
  __int64 v6; // rbx
  unsigned __int16 v7; // r12
  __int64 v8; // rax
  int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r12
  __int32 v18; // edi
  __int64 v19; // rax
  _QWORD *v20; // rsi
  unsigned __int8 *v21; // rax
  _OWORD *v22; // rdx
  __int64 v23; // r14
  int v24; // edx
  __int64 v25; // rax
  _OWORD *v26; // r12
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned __int64 v31; // rdx
  __int32 v32; // edi
  __int64 v33; // r9
  int v34; // edx
  const __m128i *v35; // rcx
  __int8 *v36; // rax
  __int64 v37; // r8
  __int32 v38; // eax
  __m128i v39; // xmm3
  __m128i v40; // xmm2
  __m128i v41; // xmm4
  __m128i v42; // xmm1
  __m128i v43; // xmm0
  _OWORD *v44; // rsi
  _OWORD *v45; // rcx
  unsigned int v46; // r14d
  int v47; // ebx
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // r8
  __m128i *v51; // rdx
  __int64 v52; // rsi
  _QWORD *v53; // r14
  __int64 v54; // r8
  unsigned int v55; // eax
  __int64 v56; // r9
  __int64 v57; // r13
  __int64 v58; // r14
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int128 v62; // [rsp-10h] [rbp-1E0h]
  __int32 v63; // [rsp+Ch] [rbp-1C4h]
  __int64 v64; // [rsp+10h] [rbp-1C0h]
  int v65; // [rsp+18h] [rbp-1B8h]
  __int32 v66; // [rsp+1Ch] [rbp-1B4h]
  __m128i v67; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v68; // [rsp+30h] [rbp-1A0h]
  __int64 v69; // [rsp+38h] [rbp-198h]
  _OWORD *v70; // [rsp+40h] [rbp-190h]
  __m128i *v71; // [rsp+48h] [rbp-188h]
  __int64 v72; // [rsp+58h] [rbp-178h]
  int v73; // [rsp+60h] [rbp-170h]
  __int64 v74; // [rsp+64h] [rbp-16Ch]
  int v75; // [rsp+6Ch] [rbp-164h]
  __int64 v76; // [rsp+70h] [rbp-160h]
  int v77; // [rsp+78h] [rbp-158h]
  __int64 v78; // [rsp+7Ch] [rbp-154h]
  int v79; // [rsp+84h] [rbp-14Ch]
  __int64 v80; // [rsp+88h] [rbp-148h]
  int v81; // [rsp+90h] [rbp-140h]
  __int64 v82; // [rsp+94h] [rbp-13Ch]
  int v83; // [rsp+9Ch] [rbp-134h]
  __int64 v84; // [rsp+A0h] [rbp-130h]
  int v85; // [rsp+A8h] [rbp-128h]
  __int64 v86; // [rsp+ACh] [rbp-124h]
  int v87; // [rsp+B4h] [rbp-11Ch]
  __int64 v88; // [rsp+B8h] [rbp-118h]
  int v89; // [rsp+C0h] [rbp-110h]
  __int64 v90; // [rsp+C4h] [rbp-10Ch]
  int v91; // [rsp+CCh] [rbp-104h]
  __m128i v92; // [rsp+D0h] [rbp-100h] BYREF
  __m128i v93; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v94; // [rsp+F0h] [rbp-E0h] BYREF
  __m128i v95; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v96; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v97; // [rsp+118h] [rbp-B8h]
  _OWORD v98[11]; // [rsp+120h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = *(_WORD *)(a2 + 96);
  v8 = *(_QWORD *)(a2 + 104);
  LOWORD(v96) = v7;
  v97 = v8;
  if ( v7 )
  {
    if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
      v7 = word_4456580[v7 - 1];
  }
  else if ( sub_30070B0((__int64)&v96) )
  {
    v7 = sub_3009970((__int64)&v96, a2, v11, v12, v13);
    v9 = *(_DWORD *)(a2 + 24);
    if ( a3 )
      goto LABEL_5;
    goto LABEL_8;
  }
  v9 = *(_DWORD *)(a2 + 24);
  if ( a3 )
  {
LABEL_5:
    v73 = 3678;
    v72 = 0xE5D00000E5CLL;
    v10 = v9 - 565;
    v74 = 0xE5400000E53LL;
    v76 = 0xE5700000E56LL;
    v78 = 0xE5A00000E59LL;
    v75 = 3669;
    v77 = 3672;
    v79 = 3675;
    v80 = 0xE4E00000E4DLL;
    v81 = 3663;
    v82 = 0xE5100000E50LL;
    v83 = 3666;
    switch ( v7 )
    {
      case 5u:
        v65 = *((_DWORD *)&v72 + v10);
        goto LABEL_10;
      case 6u:
        v65 = *((_DWORD *)&v74 + v10);
        goto LABEL_10;
      case 7u:
        v65 = *((_DWORD *)&v76 + v10);
        goto LABEL_10;
      case 8u:
        v65 = *((_DWORD *)&v78 + v10);
        goto LABEL_10;
      case 0xCu:
        v65 = *((_DWORD *)&v80 + v10);
        goto LABEL_10;
      case 0xDu:
        v65 = *((_DWORD *)&v82 + v10);
        goto LABEL_10;
      default:
        goto LABEL_43;
    }
  }
LABEL_8:
  v85 = 3696;
  v84 = 0xE6F00000E6ELL;
  v14 = v9 - 559;
  v86 = 0xE6600000E65LL;
  v88 = 0xE6900000E68LL;
  v90 = 0xE6C00000E6BLL;
  v87 = 3687;
  v89 = 3690;
  v91 = 3693;
  v92.m128i_i64[0] = 0xE6000000E5FLL;
  v92.m128i_i32[2] = 3681;
  v96 = 0xE6300000E62LL;
  LODWORD(v97) = 3684;
  switch ( v7 )
  {
    case 5u:
      v65 = *((_DWORD *)&v84 + v14);
      break;
    case 6u:
      v65 = *((_DWORD *)&v86 + v14);
      break;
    case 7u:
      v65 = *((_DWORD *)&v88 + v14);
      break;
    case 8u:
      v65 = *((_DWORD *)&v90 + v14);
      break;
    case 0xCu:
      v65 = v92.m128i_i32[v14];
      break;
    case 0xDu:
      v65 = *((_DWORD *)&v96 + v14);
      break;
    default:
LABEL_43:
      BUG();
  }
LABEL_10:
  v15 = *(_QWORD *)(a2 + 40);
  v16 = *(_QWORD *)(a2 + 80);
  v17 = *(_QWORD *)(a1 + 64);
  v64 = *(_QWORD *)v15;
  v18 = *(_DWORD *)(v15 + 8);
  v96 = v16;
  v66 = v18;
  if ( v16 )
  {
    sub_B96E90((__int64)&v96, v16, 1);
    v15 = *(_QWORD *)(a2 + 40);
  }
  LODWORD(v97) = *(_DWORD *)(a2 + 72);
  v19 = *(_QWORD *)(*(_QWORD *)(v15 + 40) + 96LL);
  v20 = *(_QWORD **)(v19 + 24);
  if ( *(_DWORD *)(v19 + 32) > 0x40u )
    v20 = (_QWORD *)*v20;
  v21 = sub_3400BD0(v17, (__int64)v20, (__int64)&v96, 8, 0, 1u, a4, 0);
  v70 = v22;
  v23 = (__int64)v21;
  if ( v96 )
    sub_B91220((__int64)&v96, v96);
  v24 = *(_DWORD *)(a2 + 64);
  v25 = *(_QWORD *)(a2 + 40);
  v26 = v98;
  v27 = (unsigned int)(v24 - 3);
  v28 = 5LL * (unsigned int)(v24 - 2);
  v29 = v25 + 40 * v27;
  v30 = *(_QWORD *)(v25 + 8 * v28);
  v31 = *(_QWORD *)(v25 + 8 * v28 + 8);
  v32 = *(_DWORD *)(v29 + 8);
  v33 = *(_QWORD *)v29;
  v92.m128i_i64[0] = 0;
  v71 = &v92;
  LODWORD(v68) = v32;
  v67.m128i_i64[0] = v33;
  v92.m128i_i32[2] = 0;
  v96 = 0;
  LODWORD(v97) = 0;
  sub_36DF750(v6, v30, v31, (__int64)&v92, (__int64)&v96, a4);
  v34 = *(_DWORD *)(a2 + 64);
  v35 = *(const __m128i **)(a2 + 40);
  v36 = &v35->m128i_i8[40 * (v34 - 1)];
  v37 = *(_QWORD *)v36;
  v38 = *((_DWORD *)v36 + 2);
  v94.m128i_i64[0] = v92.m128i_i64[0];
  v92.m128i_i64[0] = v23;
  v93.m128i_i64[0] = v67.m128i_i64[0];
  v94.m128i_i32[2] = v92.m128i_i32[2];
  v93.m128i_i32[2] = v32;
  v39 = _mm_load_si128(&v94);
  v95.m128i_i64[0] = v96;
  v40 = _mm_load_si128(&v93);
  v96 = (__int64)v98;
  v95.m128i_i32[2] = v97;
  v41 = _mm_load_si128(&v95);
  v98[1] = v40;
  v92.m128i_i32[2] = (int)v70;
  v42 = _mm_load_si128(&v92);
  v97 = 0x800000004LL;
  v98[0] = v42;
  v98[2] = v39;
  v98[3] = v41;
  if ( v34 == 5 )
  {
    v92.m128i_i32[2] = v38;
    v49 = 4;
    v92.m128i_i64[0] = v37;
    v93.m128i_i64[0] = v64;
    v93.m128i_i32[2] = v66;
  }
  else
  {
    v43 = _mm_loadu_si128(v35 + 5);
    v44 = v98;
    v45 = v98;
    v68 = v6;
    v46 = 3;
    v47 = v34 - 3;
    v48 = v37;
    v49 = 4;
    while ( 1 )
    {
      v45[v49] = v43;
      v49 = (unsigned int)(v97 + 1);
      LODWORD(v97) = v97 + 1;
      if ( v47 == v46 )
        break;
      v43 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * v46));
      if ( v49 + 1 > (unsigned __int64)HIDWORD(v97) )
      {
        v63 = v38;
        v70 = v44;
        v67 = v43;
        sub_C8D5F0((__int64)&v96, v44, v49 + 1, 0x10u, v37, v49 + 1);
        v49 = (unsigned int)v97;
        v38 = v63;
        v43 = _mm_load_si128(&v67);
        v44 = v70;
      }
      v45 = (_OWORD *)v96;
      ++v46;
    }
    v92.m128i_i32[2] = v38;
    v50 = v48;
    v92.m128i_i64[0] = v48;
    v26 = v44;
    v93.m128i_i64[0] = v64;
    v6 = v68;
    v93.m128i_i32[2] = v66;
    if ( HIDWORD(v97) < (unsigned __int64)(v49 + 2) )
    {
      sub_C8D5F0((__int64)&v96, v44, v49 + 2, 0x10u, v50, v49 + 2);
      v49 = (unsigned int)v97;
    }
  }
  v51 = (__m128i *)(v96 + 16 * v49);
  *v51 = _mm_load_si128(&v92);
  v51[1] = _mm_load_si128(&v93);
  v52 = *(_QWORD *)(a2 + 80);
  v53 = *(_QWORD **)(v6 + 64);
  v54 = v96;
  v55 = v97 + 2;
  v92.m128i_i64[0] = v52;
  LODWORD(v97) = v55;
  v56 = v55;
  if ( v52 )
  {
    v68 = v96;
    v69 = v55;
    sub_B96E90((__int64)v71, v52, 1);
    v54 = v68;
    v56 = v69;
  }
  *((_QWORD *)&v62 + 1) = v56;
  v57 = (__int64)v71;
  *(_QWORD *)&v62 = v54;
  v92.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  v58 = sub_33F7800(v53, v65, (__int64)v71, 1u, 0, v56, v62);
  sub_34158F0(*(_QWORD *)(v6 + 64), a2, v58, v59, v60, v61);
  sub_3421DB0(v58);
  sub_33ECEA0(*(const __m128i **)(v6 + 64), a2);
  if ( v92.m128i_i64[0] )
    sub_B91220(v57, v92.m128i_i64[0]);
  if ( (_OWORD *)v96 != v26 )
    _libc_free(v96);
}
