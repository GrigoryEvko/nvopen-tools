// Function: sub_303AE60
// Address: 0x303ae60
//
__int64 __fastcall sub_303AE60(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rsi
  unsigned __int16 *v10; // rdx
  __int64 v11; // rbx
  __int128 v12; // xmm1
  __int128 v13; // rax
  int v14; // r9d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r14
  __int128 v19; // rax
  int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // r14
  __int128 v25; // rax
  int v26; // r9d
  __int128 v27; // rax
  int v28; // r9d
  __int128 v29; // rax
  int v30; // r9d
  __int128 v31; // rax
  int v32; // r9d
  __int128 v33; // rax
  __int64 *v34; // r14
  __int64 (__fastcall *v35)(__int64, __int64, __int64 *, __int64, __int64); // r15
  __int64 v36; // rax
  int v37; // r9d
  __int16 v38; // r10
  __int64 v39; // r14
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // r15
  __int128 v43; // rax
  __int128 v44; // rax
  int v45; // r9d
  unsigned int v46; // edx
  __int64 v47; // r14
  __int64 v48; // rdx
  __int64 v49; // r15
  __int128 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r15
  __int64 v54; // r14
  int v55; // r9d
  __int128 v56; // rax
  int v57; // r9d
  __int64 v58; // r14
  int v60; // ecx
  __int16 v61; // ax
  int v62; // edx
  int v63; // edx
  __int128 v64; // [rsp-40h] [rbp-F0h]
  __int128 v65; // [rsp-30h] [rbp-E0h]
  __int128 v66; // [rsp-30h] [rbp-E0h]
  __int128 v67; // [rsp-20h] [rbp-D0h]
  __int128 v68; // [rsp-20h] [rbp-D0h]
  unsigned __int16 v69; // [rsp+0h] [rbp-B0h]
  int v70; // [rsp+0h] [rbp-B0h]
  __int16 v71; // [rsp+8h] [rbp-A8h]
  unsigned int v72; // [rsp+8h] [rbp-A8h]
  __int128 v73; // [rsp+10h] [rbp-A0h]
  __int128 v74; // [rsp+20h] [rbp-90h]
  int v75; // [rsp+40h] [rbp-70h]
  int v76; // [rsp+40h] [rbp-70h]
  unsigned int v77; // [rsp+48h] [rbp-68h]
  __int64 v78; // [rsp+60h] [rbp-50h] BYREF
  int v79; // [rsp+68h] [rbp-48h]
  unsigned __int16 v80; // [rsp+70h] [rbp-40h] BYREF
  __int64 v81; // [rsp+78h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v78 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v78, v9, 1);
  v10 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v79 = *(_DWORD *)(a2 + 72);
  v11 = *((_QWORD *)v10 + 1);
  v12 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v69 = *v10;
  v77 = v69;
  *(_QWORD *)&v13 = sub_33FAF80(a4, 245, (unsigned int)&v78, *v10, v11, a6, v12);
  v73 = v13;
  v15 = sub_33FAF80(a4, 234, (unsigned int)&v78, 7, 0, v14, v12);
  v17 = v16;
  v18 = v15;
  *(_QWORD *)&v19 = sub_3400BD0(a4, 0x80000000, (unsigned int)&v78, 7, 0, 0, 0);
  *((_QWORD *)&v65 + 1) = v17;
  *(_QWORD *)&v65 = v18;
  v21 = sub_3406EB0(a4, 186, (unsigned int)&v78, 7, 0, v20, v65, v19);
  v23 = v22;
  v24 = v21;
  *(_QWORD *)&v25 = sub_3400BD0(a4, 1056964608, (unsigned int)&v78, 7, 0, 0, 0);
  *((_QWORD *)&v66 + 1) = v23;
  *(_QWORD *)&v66 = v24;
  *(_QWORD *)&v27 = sub_3406EB0(a4, 187, (unsigned int)&v78, 7, 0, v26, v66, v25);
  *(_QWORD *)&v29 = sub_33FAF80(a4, 234, (unsigned int)&v78, v69, v11, v28, v27);
  *(_QWORD *)&v31 = sub_3406EB0(a4, 96, (unsigned int)&v78, v69, v11, v30, v12, v29);
  *(_QWORD *)&v33 = sub_33FAF80(a4, 269, (unsigned int)&v78, v69, v11, v32, v31);
  v34 = *(__int64 **)(a4 + 64);
  v74 = v33;
  v35 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)a1 + 528LL);
  v36 = sub_2E79000(*(__int64 **)(a4 + 40));
  if ( v35 != sub_30368A0 )
  {
    v75 = v35(a1, v36, v34, v69, v11);
    v38 = v75;
    v37 = v63;
    goto LABEL_7;
  }
  v81 = v11;
  v80 = v69;
  if ( v69 )
  {
    if ( (unsigned __int16)(v69 - 17) > 0xD3u )
      goto LABEL_6;
    if ( (unsigned __int16)(v69 - 176) > 0x34u )
      goto LABEL_12;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v80) )
    {
LABEL_6:
      v37 = 0;
      v38 = 2;
      goto LABEL_7;
    }
    if ( !sub_3007100((__int64)&v80) )
      goto LABEL_17;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v80 )
  {
    if ( (unsigned __int16)(v80 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_12:
    v60 = word_4456340[v80 - 1];
    goto LABEL_13;
  }
LABEL_17:
  v60 = sub_3007130((__int64)&v80, 269);
LABEL_13:
  v72 = v60;
  v61 = sub_2D43050(2, v60);
  v37 = 0;
  v38 = v61;
  if ( !v61 )
  {
    v38 = sub_3009400(v34, 2, 0, v72, 0);
    v37 = v62;
  }
LABEL_7:
  v70 = v37;
  v71 = v38;
  v39 = sub_33FE730(a4, &v78, v77, v11, 0, 8388608.0);
  HIWORD(v40) = HIWORD(v75);
  v42 = v41;
  LOWORD(v40) = v71;
  v76 = v40;
  *(_QWORD *)&v43 = sub_33ED040(a4, 2);
  *((_QWORD *)&v67 + 1) = v42;
  *(_QWORD *)&v67 = v39;
  *(_QWORD *)&v44 = sub_340F900(a4, 208, (unsigned int)&v78, v76, v70, v70, v73, v67, v43);
  *(_QWORD *)&v74 = sub_340F900(a4, 205, (unsigned int)&v78, v77, v11, v45, v44, v12, v74);
  *((_QWORD *)&v74 + 1) = v46 | *((_QWORD *)&v74 + 1) & 0xFFFFFFFF00000000LL;
  v47 = sub_33FE730(a4, &v78, v77, v11, 0, 0.5);
  v49 = v48;
  *(_QWORD *)&v50 = sub_33ED040(a4, 4);
  *((_QWORD *)&v68 + 1) = v49;
  *(_QWORD *)&v68 = v47;
  v51 = sub_340F900(a4, 208, (unsigned int)&v78, v76, v70, v70, v73, v68, v50);
  v53 = v52;
  v54 = v51;
  *(_QWORD *)&v56 = sub_33FAF80(a4, 269, (unsigned int)&v78, v77, v11, v55, v12);
  *((_QWORD *)&v64 + 1) = v53;
  *(_QWORD *)&v64 = v54;
  v58 = sub_340F900(a4, 205, (unsigned int)&v78, v77, v11, v57, v64, v56, v74);
  if ( v78 )
    sub_B91220((__int64)&v78, v78);
  return v58;
}
