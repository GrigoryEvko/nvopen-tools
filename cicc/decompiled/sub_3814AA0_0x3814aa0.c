// Function: sub_3814AA0
// Address: 0x3814aa0
//
unsigned __int8 *__fastcall sub_3814AA0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int64 v3; // r13
  __int64 v5; // r11
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // rbx
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // r9
  int v13; // r8d
  __int64 v14; // rsi
  _QWORD *v15; // r10
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned __int8 *v20; // rcx
  unsigned int v21; // edx
  __int64 v22; // r9
  __int64 v23; // rsi
  unsigned __int64 v24; // r13
  unsigned __int16 *v25; // rax
  _QWORD *v26; // r10
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int128 v29; // rax
  __int64 v30; // r9
  int v31; // edi
  unsigned int v32; // esi
  unsigned __int8 *v33; // r12
  bool v35; // al
  __int16 v36; // ax
  __int64 v37; // rax
  __m128i v38; // xmm0
  unsigned int v39; // edx
  __int64 v40; // rdi
  __int64 v41; // rax
  char v42; // al
  bool v43; // zf
  int v44; // eax
  __int64 v45; // rdi
  __int64 v46; // rax
  int v47; // eax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int128 v50; // [rsp-20h] [rbp-120h]
  __int128 v51; // [rsp-10h] [rbp-110h]
  __int128 v52; // [rsp-10h] [rbp-110h]
  __int64 v53; // [rsp-10h] [rbp-110h]
  __int64 v54; // [rsp+8h] [rbp-F8h]
  __int64 v55; // [rsp+10h] [rbp-F0h]
  unsigned int v56; // [rsp+18h] [rbp-E8h]
  __int64 v57; // [rsp+18h] [rbp-E8h]
  __int64 *v58; // [rsp+20h] [rbp-E0h]
  __int64 *v59; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v60; // [rsp+20h] [rbp-E0h]
  _QWORD *v61; // [rsp+20h] [rbp-E0h]
  unsigned int v62; // [rsp+28h] [rbp-D8h]
  _QWORD *v63; // [rsp+28h] [rbp-D8h]
  __int64 v64; // [rsp+30h] [rbp-D0h]
  unsigned __int16 v65; // [rsp+3Eh] [rbp-C2h]
  unsigned __int8 *v66; // [rsp+60h] [rbp-A0h]
  __int64 v67; // [rsp+70h] [rbp-90h] BYREF
  int v68; // [rsp+78h] [rbp-88h]
  _QWORD v69[2]; // [rsp+80h] [rbp-80h] BYREF
  __int16 v70; // [rsp+90h] [rbp-70h]
  __int64 v71; // [rsp+98h] [rbp-68h]
  __m128i v72; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v73; // [rsp+B0h] [rbp-50h]
  __m128i v74; // [rsp+C0h] [rbp-40h]

  v5 = *a1;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = a1[1];
  if ( v6 == sub_2D56A50 )
  {
    v11 = 0;
    sub_2FE6CC0((__int64)&v72, v5, *(_QWORD *)(v10 + 64), v8, v9);
    v12 = a1;
    v65 = v72.m128i_u16[4];
    v64 = v73.m128i_i64[0];
  }
  else
  {
    v48 = v6(v5, *(_QWORD *)(v10 + 64), v8, v9);
    v12 = a1;
    v64 = v49;
    v11 = v48;
    v65 = v48;
  }
  v13 = *(_DWORD *)(a2 + 24);
  v14 = *(_QWORD *)(a2 + 80);
  v62 = v13;
  v67 = v14;
  if ( v14 )
  {
    v58 = v12;
    sub_B96E90((__int64)&v67, v14, 1);
    v13 = *(_DWORD *)(a2 + 24);
    v12 = v58;
  }
  v15 = (_QWORD *)v12[1];
  v68 = *(_DWORD *)(a2 + 72);
  if ( (unsigned int)(*(_DWORD *)(*v15 + 544LL) - 42) > 1 && v13 == 227 )
  {
    v40 = *v12;
    v41 = 1;
    if ( v65 != 1 && (!v65 || (v41 = v65, !*(_QWORD *)(v40 + 8LL * v65 + 112))) || *(_BYTE *)(v40 + 500 * v41 + 6641) )
    {
      v42 = sub_3813820(v40, 0xE2u, v65, 0, 0xE3u);
      v13 = 227;
      v43 = v42 == 0;
      v44 = 226;
      if ( v43 )
        v44 = v62;
      v62 = v44;
    }
    v3 = 0;
LABEL_30:
    if ( (unsigned int)(v13 - 101) <= 0x2F )
      goto LABEL_31;
    goto LABEL_15;
  }
  if ( v13 == 142 )
  {
    v45 = *v12;
    v46 = 1;
    if ( v65 != 1 && (!v65 || (v46 = v65, !*(_QWORD *)(v45 + 8LL * v65 + 112))) || *(_BYTE *)(v45 + 500 * v46 + 6556) )
    {
      v43 = (unsigned __int8)sub_3813820(v45, 0x8Du, v65, 0, 0x8Eu) == 0;
      v47 = 141;
      if ( v43 )
        v47 = v62;
      v62 = v47;
    }
    goto LABEL_31;
  }
  if ( v13 == 452 )
  {
    v16 = *v12;
    v17 = 1;
    if ( v65 != 1 && (!v65 || (v17 = v65, !*(_QWORD *)(v16 + 8LL * v65 + 112))) || *(_BYTE *)(v16 + 500 * v17 + 6866) )
    {
      if ( (unsigned __int8)sub_3813820(v16, 0x1C5u, v65, 0, 0x1C4u) )
      {
        v62 = 453;
        v18 = *(_QWORD *)(a2 + 40);
        goto LABEL_12;
      }
    }
    v21 = v62 - 452;
    v3 = 0;
LABEL_16:
    v18 = *(_QWORD *)(a2 + 40);
    if ( v21 > 1 )
    {
      LOWORD(v11) = v65;
      v59 = v12;
      v20 = sub_33FAF80((__int64)v15, v62, (__int64)&v67, (unsigned int)v11, v64, (_DWORD)v12, a3);
      goto LABEL_18;
    }
LABEL_12:
    LOWORD(v11) = v65;
    v59 = v12;
    v72 = _mm_loadu_si128((const __m128i *)v18);
    v73 = _mm_loadu_si128((const __m128i *)(v18 + 40));
    *((_QWORD *)&v51 + 1) = 3;
    *(_QWORD *)&v51 = &v72;
    v74 = _mm_loadu_si128((const __m128i *)(v18 + 80));
    v20 = sub_33FC220(v15, v62, (__int64)&v67, (unsigned int)v11, v64, (__int64)v12, v51);
LABEL_18:
    v22 = (__int64)v59;
    v23 = 0xFFFFFFFF00000000LL;
    v24 = v19 | v3 & 0xFFFFFFFF00000000LL;
    goto LABEL_19;
  }
  v3 = 0;
  if ( v13 <= 239 )
  {
    if ( v13 <= 237 )
      goto LABEL_30;
  }
  else if ( (unsigned int)(v13 - 242) > 1 )
  {
LABEL_15:
    v21 = v62 - 452;
    goto LABEL_16;
  }
LABEL_31:
  v37 = *(_QWORD *)(a2 + 40);
  LOWORD(v11) = v65;
  v57 = (__int64)v12;
  v38 = _mm_loadu_si128((const __m128i *)v37);
  v69[1] = v64;
  v70 = 1;
  v72 = v38;
  *((_QWORD *)&v52 + 1) = 2;
  *(_QWORD *)&v52 = &v72;
  v73 = _mm_loadu_si128((const __m128i *)(v37 + 40));
  v69[0] = v11;
  v71 = 0;
  v66 = sub_3411BE0(v15, v62, (__int64)&v67, (unsigned __int16 *)v69, 2, (__int64)v12, v52);
  v24 = v39 | v3 & 0xFFFFFFFF00000000LL;
  sub_3760E70(v57, a2, 1, (unsigned __int64)v66, 1);
  v23 = v53;
  v22 = v57;
  v20 = v66;
LABEL_19:
  v25 = *(unsigned __int16 **)(a2 + 48);
  v26 = *(_QWORD **)(v22 + 8);
  LODWORD(v27) = *v25;
  v28 = *((_QWORD *)v25 + 1);
  v72.m128i_i16[0] = v27;
  v72.m128i_i64[1] = v28;
  if ( (_WORD)v27 )
  {
    if ( (unsigned __int16)(v27 - 17) <= 0xD3u )
    {
      v28 = 0;
      LOWORD(v27) = word_4456580[(int)v27 - 1];
    }
  }
  else
  {
    v54 = (__int64)v20;
    v55 = v28;
    v56 = v27;
    v61 = v26;
    v35 = sub_30070B0((__int64)&v72);
    v26 = v61;
    LOWORD(v27) = v56;
    v28 = v55;
    v20 = (unsigned __int8 *)v54;
    if ( v35 )
    {
      v36 = sub_3009970((__int64)&v72, v23, v56, v54, v55);
      v20 = (unsigned __int8 *)v54;
      v26 = v61;
      v28 = v27;
      LOWORD(v27) = v36;
    }
  }
  v60 = v20;
  v63 = v26;
  *(_QWORD *)&v29 = sub_33F7D60(v26, (unsigned __int16)v27, v28);
  v31 = *(_DWORD *)(a2 + 24);
  if ( v31 == 142 || v31 == 227 || (v32 = 3, v31 == 452) )
    v32 = 4;
  LOWORD(v11) = v65;
  *((_QWORD *)&v50 + 1) = v24;
  *(_QWORD *)&v50 = v60;
  v33 = sub_3406EB0(v63, v32, (__int64)&v67, (unsigned int)v11, v64, v30, v50, v29);
  if ( v67 )
    sub_B91220((__int64)&v67, v67);
  return v33;
}
