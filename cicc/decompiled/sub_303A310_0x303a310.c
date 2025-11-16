// Function: sub_303A310
// Address: 0x303a310
//
__int64 __fastcall sub_303A310(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rdx
  unsigned __int16 v7; // ax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // rsi
  unsigned __int32 v15; // r10d
  bool v16; // zf
  __int64 *v17; // rax
  int v18; // r13d
  __int64 v19; // r14
  __int64 v20; // r15
  __int128 v21; // rax
  int v22; // r9d
  __m128i v23; // rax
  int v24; // r9d
  __int128 v25; // rax
  __int128 v26; // rax
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // r14
  int v32; // r9d
  __int128 v33; // rax
  int v34; // r9d
  __m128i v35; // rax
  int v36; // r9d
  __int128 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // r15
  __int128 v41; // rax
  int v42; // r9d
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r15
  __int64 v46; // r14
  int v47; // r9d
  __m128i v48; // rax
  int v49; // r9d
  __int64 v50; // rax
  __m128i v51; // xmm2
  __int64 v52; // rdx
  __int64 v53; // r12
  __int64 v55; // rax
  __m128i v56; // rax
  int v57; // r9d
  __int64 v58; // rax
  __m128i v59; // xmm3
  __int64 v60; // rdx
  __int128 v61; // [rsp-30h] [rbp-100h]
  __int128 v62; // [rsp-30h] [rbp-100h]
  __int128 v63; // [rsp-20h] [rbp-F0h]
  __int128 v64; // [rsp-20h] [rbp-F0h]
  __int128 v65; // [rsp-10h] [rbp-E0h]
  __int128 v66; // [rsp+0h] [rbp-D0h]
  __int128 v67; // [rsp+0h] [rbp-D0h]
  __m128i v68; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+20h] [rbp-B0h]
  __int64 *v70; // [rsp+28h] [rbp-A8h]
  __int128 v71; // [rsp+30h] [rbp-A0h]
  __m128i v72; // [rsp+40h] [rbp-90h] BYREF
  int v73; // [rsp+50h] [rbp-80h] BYREF
  __int64 v74; // [rsp+58h] [rbp-78h]
  __int64 v75; // [rsp+60h] [rbp-70h] BYREF
  int v76; // [rsp+68h] [rbp-68h]
  __int64 v77; // [rsp+70h] [rbp-60h]
  __int64 v78; // [rsp+78h] [rbp-58h]
  __int64 v79; // [rsp+80h] [rbp-50h] BYREF
  __int64 v80; // [rsp+88h] [rbp-48h]
  __m128i v81; // [rsp+90h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v68.m128i_i64[0] = a1;
  v7 = *(_WORD *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  LOWORD(v73) = v7;
  v74 = v8;
  if ( v7 )
  {
    if ( v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v55 = 16LL * (v7 - 1);
    v10 = *(_QWORD *)&byte_444C4A0[v55];
    v11 = byte_444C4A0[v55 + 8];
  }
  else
  {
    v77 = sub_3007260((__int64)&v73);
    v78 = v9;
    v10 = v77;
    v11 = v78;
  }
  LOBYTE(v80) = v11;
  v79 = v10;
  v70 = &v79;
  v12 = sub_CA1930(&v79);
  v14 = *(_QWORD *)(a2 + 80);
  v15 = v12;
  v75 = v14;
  if ( v14 )
  {
    v72.m128i_i64[0] = v12;
    sub_B96E90((__int64)&v75, v14, 1);
    v15 = v72.m128i_i32[0];
  }
  v16 = *(_DWORD *)(a2 + 24) == 211;
  v76 = *(_DWORD *)(a2 + 72);
  v17 = *(__int64 **)(a2 + 40);
  v18 = !v16 + 191;
  v19 = *v17;
  v20 = v17[1];
  v71 = (__int128)_mm_loadu_si128((const __m128i *)(v17 + 5));
  v72 = _mm_loadu_si128((const __m128i *)v17 + 5);
  if ( v15 == 32 && *(_DWORD *)(*(_QWORD *)(v68.m128i_i64[0] + 537016) + 340LL) > 0x15Du )
  {
    v56.m128i_i64[0] = sub_3406EB0(a4, v18, (unsigned int)&v75, v73, v74, v13, v71, *(_OWORD *)&v72);
    *((_QWORD *)&v64 + 1) = v20;
    *(_QWORD *)&v64 = v19;
    v68 = v56;
    v58 = sub_340F900(a4, 530, (unsigned int)&v75, v73, v74, v57, v71, v64, *(_OWORD *)&v72);
    v59 = _mm_load_si128(&v68);
    v79 = v58;
    v80 = v60;
    v81 = v59;
  }
  else
  {
    v69 = v15;
    *(_QWORD *)&v21 = sub_3400BD0(a4, v15, (unsigned int)&v75, 7, 0, 0, 0);
    v23.m128i_i64[0] = sub_3406EB0(a4, 57, (unsigned int)&v75, 7, 0, v22, v21, *(_OWORD *)&v72);
    *((_QWORD *)&v63 + 1) = v20;
    *(_QWORD *)&v63 = v19;
    v68 = v23;
    *(_QWORD *)&v25 = sub_3406EB0(a4, 192, (unsigned int)&v75, v73, v74, v24, v63, *(_OWORD *)&v72);
    v66 = v25;
    *(_QWORD *)&v26 = sub_3400BD0(a4, v69, (unsigned int)&v75, 7, 0, 0, 0);
    v28 = sub_3406EB0(a4, 57, (unsigned int)&v75, 7, 0, v27, *(_OWORD *)&v72, v26);
    v30 = v29;
    v31 = v28;
    *(_QWORD *)&v33 = sub_3406EB0(a4, 190, (unsigned int)&v75, v73, v74, v32, v71, *(_OWORD *)&v68);
    v35.m128i_i64[0] = sub_3406EB0(a4, 187, (unsigned int)&v75, v73, v74, v34, v66, v33);
    *((_QWORD *)&v65 + 1) = v30;
    *(_QWORD *)&v65 = v31;
    v68 = v35;
    *(_QWORD *)&v37 = sub_3406EB0(a4, v18, (unsigned int)&v75, v73, v74, v36, v71, v65);
    v67 = v37;
    v38 = sub_3400BD0(a4, v69, (unsigned int)&v75, 7, 0, 0, 0);
    v40 = v39;
    *(_QWORD *)&v41 = sub_33ED040(a4, 19);
    *((_QWORD *)&v61 + 1) = v40;
    *(_QWORD *)&v61 = v38;
    v43 = sub_340F900(a4, 208, (unsigned int)&v75, 2, 0, v42, *(_OWORD *)&v72, v61, v41);
    v45 = v44;
    v46 = v43;
    v48.m128i_i64[0] = sub_3406EB0(a4, v18, (unsigned int)&v75, v73, v74, v47, v71, *(_OWORD *)&v72);
    *((_QWORD *)&v62 + 1) = v45;
    *(_QWORD *)&v62 = v46;
    v72 = v48;
    v50 = sub_340F900(a4, 205, (unsigned int)&v75, v73, v74, v49, v62, v67, *(_OWORD *)&v68);
    v51 = _mm_load_si128(&v72);
    v79 = v50;
    v80 = v52;
    v81 = v51;
  }
  v53 = sub_3411660(a4, v70, 2, &v75);
  if ( v75 )
    sub_B91220((__int64)&v75, v75);
  return v53;
}
