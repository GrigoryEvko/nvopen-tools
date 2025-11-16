// Function: sub_303A7B0
// Address: 0x303a7b0
//
__int64 __fastcall sub_303A7B0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rdx
  unsigned __int16 v7; // ax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int8 v11; // al
  int v12; // eax
  int v13; // r9d
  __int64 v14; // rsi
  int v15; // r14d
  const __m128i *v16; // rax
  __int64 v17; // r12
  __int64 v18; // r13
  __int128 v19; // rax
  int v20; // r9d
  __int128 v21; // rax
  int v22; // r9d
  __int128 v23; // rax
  __int128 v24; // rax
  int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // r12
  int v30; // r9d
  __int128 v31; // rax
  int v32; // r9d
  __int128 v33; // rax
  int v34; // r9d
  __int128 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rdx
  __int64 v38; // r13
  __int128 v39; // rax
  int v40; // r9d
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r13
  __int64 v44; // r12
  int v45; // r9d
  __m128i v46; // rax
  int v47; // r9d
  __int64 v48; // rax
  __m128i v49; // xmm2
  __int64 v50; // rdx
  __int64 v51; // r12
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r13
  __int64 v57; // r12
  int v58; // r9d
  __m128i v59; // rax
  __int128 v60; // [rsp-30h] [rbp-100h]
  __int128 v61; // [rsp-30h] [rbp-100h]
  __int128 v62; // [rsp-30h] [rbp-100h]
  __int128 v63; // [rsp-20h] [rbp-F0h]
  __int128 v64; // [rsp-10h] [rbp-E0h]
  __int128 v65; // [rsp+0h] [rbp-D0h]
  __int128 v66; // [rsp+0h] [rbp-D0h]
  __int128 v67; // [rsp+10h] [rbp-C0h]
  __int128 v68; // [rsp+10h] [rbp-C0h]
  __int128 v69; // [rsp+30h] [rbp-A0h]
  __m128i v70; // [rsp+40h] [rbp-90h] BYREF
  int v71; // [rsp+50h] [rbp-80h] BYREF
  __int64 v72; // [rsp+58h] [rbp-78h]
  __int64 v73; // [rsp+60h] [rbp-70h] BYREF
  int v74; // [rsp+68h] [rbp-68h]
  __int64 v75; // [rsp+70h] [rbp-60h]
  __int64 v76; // [rsp+78h] [rbp-58h]
  __m128i v77; // [rsp+80h] [rbp-50h] BYREF
  __int64 v78; // [rsp+90h] [rbp-40h]
  __int64 v79; // [rsp+98h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v7 = *(_WORD *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  LOWORD(v71) = v7;
  v72 = v8;
  if ( v7 )
  {
    if ( v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v53 = 16LL * (v7 - 1);
    v10 = *(_QWORD *)&byte_444C4A0[v53];
    v11 = byte_444C4A0[v53 + 8];
  }
  else
  {
    v75 = sub_3007260((__int64)&v71);
    v76 = v9;
    v10 = v75;
    v11 = v76;
  }
  v77.m128i_i8[8] = v11;
  v77.m128i_i64[0] = v10;
  v12 = sub_CA1930(&v77);
  v14 = *(_QWORD *)(a2 + 80);
  v15 = v12;
  v73 = v14;
  if ( v14 )
    sub_B96E90((__int64)&v73, v14, 1);
  v74 = *(_DWORD *)(a2 + 72);
  v16 = *(const __m128i **)(a2 + 40);
  v17 = v16[2].m128i_i64[1];
  v18 = v16[3].m128i_i64[0];
  v69 = (__int128)_mm_loadu_si128(v16);
  v70 = _mm_loadu_si128(v16 + 5);
  if ( v15 == 32 && *(_DWORD *)(*(_QWORD *)(a1 + 537016) + 340LL) > 0x15Du )
  {
    *((_QWORD *)&v62 + 1) = v18;
    *(_QWORD *)&v62 = v17;
    v54 = sub_340F900(a4, 529, (unsigned int)&v73, v71, v72, v13, v62, v69, *(_OWORD *)&v70);
    v56 = v55;
    v57 = v54;
    v59.m128i_i64[0] = sub_3406EB0(a4, 190, (unsigned int)&v73, v71, v72, v58, v69, *(_OWORD *)&v70);
    v78 = v57;
    v77 = v59;
    v79 = v56;
  }
  else
  {
    *(_QWORD *)&v19 = sub_3400BD0(a4, v15, (unsigned int)&v73, 7, 0, 0, 0);
    *(_QWORD *)&v21 = sub_3406EB0(a4, 57, (unsigned int)&v73, 7, 0, v20, v19, *(_OWORD *)&v70);
    *((_QWORD *)&v63 + 1) = v18;
    *(_QWORD *)&v63 = v17;
    v65 = v21;
    *(_QWORD *)&v23 = sub_3406EB0(a4, 190, (unsigned int)&v73, v71, v72, v22, v63, *(_OWORD *)&v70);
    v67 = v23;
    *(_QWORD *)&v24 = sub_3400BD0(a4, v15, (unsigned int)&v73, 7, 0, 0, 0);
    v26 = sub_3406EB0(a4, 57, (unsigned int)&v73, 7, 0, v25, *(_OWORD *)&v70, v24);
    v28 = v27;
    v29 = v26;
    *(_QWORD *)&v31 = sub_3406EB0(a4, 192, (unsigned int)&v73, v71, v72, v30, v69, v65);
    *(_QWORD *)&v33 = sub_3406EB0(a4, 187, (unsigned int)&v73, v71, v72, v32, v67, v31);
    *((_QWORD *)&v64 + 1) = v28;
    *(_QWORD *)&v64 = v29;
    v68 = v33;
    *(_QWORD *)&v35 = sub_3406EB0(a4, 190, (unsigned int)&v73, v71, v72, v34, v69, v64);
    v66 = v35;
    v36 = sub_3400BD0(a4, v15, (unsigned int)&v73, 7, 0, 0, 0);
    v38 = v37;
    *(_QWORD *)&v39 = sub_33ED040(a4, 19);
    *((_QWORD *)&v60 + 1) = v38;
    *(_QWORD *)&v60 = v36;
    v41 = sub_340F900(a4, 208, (unsigned int)&v73, 2, 0, v40, *(_OWORD *)&v70, v60, v39);
    v43 = v42;
    v44 = v41;
    v46.m128i_i64[0] = sub_3406EB0(a4, 190, (unsigned int)&v73, v71, v72, v45, v69, *(_OWORD *)&v70);
    *((_QWORD *)&v61 + 1) = v43;
    *(_QWORD *)&v61 = v44;
    v70 = v46;
    v48 = sub_340F900(a4, 205, (unsigned int)&v73, v71, v72, v47, v61, v66, v68);
    v49 = _mm_load_si128(&v70);
    v78 = v48;
    v79 = v50;
    v77 = v49;
  }
  v51 = sub_3411660(a4, &v77, 2, &v73);
  if ( v73 )
    sub_B91220((__int64)&v73, v73);
  return v51;
}
