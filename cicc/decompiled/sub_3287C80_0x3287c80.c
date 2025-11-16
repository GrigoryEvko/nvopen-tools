// Function: sub_3287C80
// Address: 0x3287c80
//
__int64 __fastcall sub_3287C80(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  const __m128i *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r11
  unsigned __int32 v8; // r12d
  bool v9; // zf
  __int64 v10; // r10
  __int64 v11; // rcx
  __int32 v12; // r15d
  __int64 v13; // r13
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int16 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int16 *v25; // rdx
  int v26; // eax
  __int64 v27; // rdx
  __int16 v28; // ax
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int16 v33; // ax
  int v34; // r15d
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r8
  int v39; // eax
  int v40; // eax
  unsigned __int16 *v41; // rax
  unsigned int v42; // edx
  __int64 v43; // rdi
  __int64 v44; // r13
  __int16 v45; // bx
  char v46; // al
  __int64 v47; // rdx
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // r14
  unsigned int v52; // edx
  int v53; // [rsp+0h] [rbp-180h]
  int v54; // [rsp+8h] [rbp-178h]
  __m128i v55; // [rsp+10h] [rbp-170h]
  __m128i v56; // [rsp+20h] [rbp-160h]
  __int64 v57; // [rsp+30h] [rbp-150h]
  __int64 v58; // [rsp+38h] [rbp-148h]
  __int64 v59; // [rsp+40h] [rbp-140h]
  int v60; // [rsp+48h] [rbp-138h]
  unsigned int v61; // [rsp+48h] [rbp-138h]
  int v62; // [rsp+50h] [rbp-130h]
  __int64 v63; // [rsp+50h] [rbp-130h]
  __int64 v64; // [rsp+58h] [rbp-128h]
  unsigned int v65; // [rsp+60h] [rbp-120h]
  int v66; // [rsp+60h] [rbp-120h]
  unsigned __int8 v67; // [rsp+60h] [rbp-120h]
  int v68; // [rsp+60h] [rbp-120h]
  __int64 v69; // [rsp+68h] [rbp-118h]
  int v70; // [rsp+68h] [rbp-118h]
  unsigned int v71; // [rsp+68h] [rbp-118h]
  int v72; // [rsp+68h] [rbp-118h]
  unsigned __int8 (__fastcall *v73)(__int64, _QWORD, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, int *, __int64); // [rsp+70h] [rbp-110h]
  __int64 v74; // [rsp+70h] [rbp-110h]
  __int64 v75; // [rsp+70h] [rbp-110h]
  __int64 v76; // [rsp+78h] [rbp-108h]
  int v77; // [rsp+ACh] [rbp-D4h] BYREF
  __int64 v78; // [rsp+B0h] [rbp-D0h] BYREF
  int v79; // [rsp+B8h] [rbp-C8h]
  __m128i v80; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned __int16 v81; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v82; // [rsp+D8h] [rbp-A8h]
  __int64 v83; // [rsp+E0h] [rbp-A0h]
  __int64 v84; // [rsp+E8h] [rbp-98h]
  __int64 v85; // [rsp+F0h] [rbp-90h]
  __int64 v86; // [rsp+F8h] [rbp-88h]
  __int64 v87; // [rsp+100h] [rbp-80h]
  __int64 v88; // [rsp+108h] [rbp-78h]
  _BYTE v89[24]; // [rsp+110h] [rbp-70h] BYREF
  _QWORD v90[10]; // [rsp+130h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v78 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v78, v4, 1);
  v79 = *(_DWORD *)(a2 + 72);
  v5 = *(const __m128i **)(a2 + 40);
  v6 = v5[2].m128i_i64[1];
  v7 = v5[5].m128i_i64[0];
  v8 = v5[3].m128i_u32[0];
  v9 = *(_DWORD *)(v6 + 24) == 157;
  v10 = v5[5].m128i_i64[1];
  v11 = v7;
  v12 = v5[5].m128i_i32[2];
  v80 = _mm_loadu_si128(v5);
  if ( !v9 )
    goto LABEL_4;
  v15 = *(_QWORD *)(v6 + 56);
  if ( !v15 )
    goto LABEL_4;
  v16 = 1;
  do
  {
    while ( *(_DWORD *)(v15 + 8) != v8 )
    {
      v15 = *(_QWORD *)(v15 + 32);
      if ( !v15 )
        goto LABEL_16;
    }
    if ( !v16 )
      goto LABEL_4;
    v17 = *(_QWORD *)(v15 + 32);
    if ( !v17 )
      goto LABEL_17;
    if ( *(_DWORD *)(v17 + 8) == v8 )
      goto LABEL_4;
    v15 = *(_QWORD *)(v17 + 32);
    v16 = 0;
  }
  while ( v15 );
LABEL_16:
  if ( v16 == 1 )
    goto LABEL_4;
LABEL_17:
  v18 = *(_QWORD *)(v6 + 40);
  v59 = *(_QWORD *)(v18 + 80);
  v57 = *(_QWORD *)(v18 + 40);
  v56 = _mm_loadu_si128((const __m128i *)(v18 + 40));
  v58 = 16LL * *(unsigned int *)(v18 + 48);
  v19 = (unsigned __int16 *)(*(_QWORD *)(v57 + 48) + v58);
  v20 = *v19;
  v21 = *((_QWORD *)v19 + 1);
  v55 = _mm_loadu_si128((const __m128i *)(v18 + 80));
  v81 = v20;
  v82 = v21;
  if ( (_WORD)v20 )
  {
    if ( (_WORD)v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
      BUG();
    v23 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v20 - 16];
    if ( !v23 )
      goto LABEL_4;
  }
  else
  {
    v60 = v10;
    v62 = v7;
    v65 = v20;
    v69 = v7;
    v85 = sub_3007260((__int64)&v81);
    v86 = v22;
    if ( !v85 )
    {
LABEL_4:
      v13 = 0;
      goto LABEL_5;
    }
    v23 = sub_3007260((__int64)&v81);
    LODWORD(v10) = v60;
    LODWORD(v7) = v62;
    v87 = v23;
    v88 = v24;
    v20 = v65;
    v11 = v69;
  }
  if ( (v23 & 7) != 0 )
    goto LABEL_4;
  v25 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v18 + 48LL) + 16LL * *(unsigned int *)(v18 + 8));
  v26 = *v25;
  v27 = *((_QWORD *)v25 + 1);
  LOWORD(v90[0]) = v26;
  v90[1] = v27;
  if ( (_WORD)v26 )
  {
    v28 = word_4456580[v26 - 1];
    v29 = 0;
  }
  else
  {
    v68 = v10;
    v72 = v7;
    v75 = v11;
    v28 = sub_3009970((__int64)v90, v18, v27, v11, v20);
    LOWORD(v20) = v81;
    LODWORD(v10) = v68;
    LODWORD(v7) = v72;
    v11 = v75;
  }
  if ( v28 != (_WORD)v20 || !v28 && v82 != v29 )
    goto LABEL_4;
  v30 = *(__int64 **)(v6 + 40);
  v31 = *v30;
  if ( *(_DWORD *)(*v30 + 24) != 298 )
    goto LABEL_4;
  v32 = *(_QWORD *)(v31 + 40);
  if ( v11 != *(_QWORD *)(v32 + 40) )
    goto LABEL_4;
  if ( v12 != *(_DWORD *)(v32 + 48) )
    goto LABEL_4;
  v33 = *(_WORD *)(a2 + 96);
  if ( v33 != *(_WORD *)(v31 + 96) || *(_QWORD *)(v31 + 104) != *(_QWORD *)(a2 + 104) && !v33 )
    goto LABEL_4;
  v66 = v10;
  v70 = v7;
  if ( !(unsigned __int8)sub_3287C60(a2) )
    goto LABEL_4;
  if ( *(_DWORD *)(a2 + 24) != 299 )
    goto LABEL_4;
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
    goto LABEL_4;
  if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    goto LABEL_4;
  v34 = sub_2EAC1E0(*(_QWORD *)(v31 + 112));
  if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(a2 + 112)) != v34 )
    goto LABEL_4;
  if ( !(unsigned __int8)sub_33CFB90(&v80, v31, 1, 2) )
    goto LABEL_4;
  v35 = a1[1];
  v36 = *(_QWORD *)(a2 + 112);
  v53 = v66;
  v54 = v70;
  v71 = *(unsigned __int16 *)(v36 + 32);
  v73 = *(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, int *, __int64))(*(_QWORD *)v35 + 824LL);
  v67 = sub_2EAC4F0(v36);
  v61 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
  v63 = *(_QWORD *)(*(_QWORD *)(v57 + 48) + v58 + 8);
  v64 = *(unsigned __int16 *)(*(_QWORD *)(v57 + 48) + v58);
  v37 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  if ( !v73(v35, *(_QWORD *)(*a1 + 64LL), v37, v64, v63, v61, v67, v71, &v77, v38) || !v77 )
    goto LABEL_4;
  v39 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
  v89[20] = 0;
  *(_DWORD *)&v89[16] = v39;
  *(_QWORD *)v89 = 0;
  *(_QWORD *)&v89[8] = 0;
  v40 = *(_DWORD *)(v59 + 24);
  if ( v40 == 35 || v40 == 11 )
  {
    v83 = sub_2D5B750(&v81);
    v84 = v47;
    v48 = sub_325F4E0(*(_QWORD *)(*(_QWORD *)(v59 + 96) + 24LL), *(_DWORD *)(*(_QWORD *)(v59 + 96) + 32LL));
    v90[0] = v83 * v48;
    LOBYTE(v90[1]) = v84;
    v49 = sub_CA1930(v90);
    v50 = *a1;
    v49 >>= 3;
    v51 = (unsigned int)v49;
    LOBYTE(v84) = 0;
    v83 = (unsigned int)v49;
    v74 = sub_3409320(v50, v54, v53, v49, v84, (unsigned int)&v78, 0);
    v76 = v52;
    sub_327C6E0((__int64)v90, *(__int64 **)(a2 + 112), v51);
    qmemcpy(v89, v90, 0x15u);
  }
  else
  {
    v41 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * v8);
    v74 = sub_3466750(a1[1], *a1, v54, v53, *v41, *((_QWORD *)v41 + 1), v55.m128i_i64[0], v55.m128i_i64[1]);
    v76 = v42;
  }
  v43 = *(_QWORD *)(a2 + 112);
  memset(v90, 0, 32);
  v44 = *a1;
  v45 = *(_WORD *)(v43 + 32);
  v46 = sub_2EAC4F0(v43);
  v13 = sub_33F4560(
          v44,
          v80.m128i_i32[0],
          v80.m128i_i32[2],
          (unsigned int)&v78,
          v56.m128i_i32[0],
          v56.m128i_i32[2],
          v74,
          v76,
          *(__int128 *)v89,
          *(__int64 *)&v89[16],
          v46,
          v45,
          (__int64)v90);
LABEL_5:
  if ( v78 )
    sub_B91220((__int64)&v78, v78);
  return v13;
}
