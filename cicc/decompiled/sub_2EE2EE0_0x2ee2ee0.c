// Function: sub_2EE2EE0
// Address: 0x2ee2ee0
//
__int64 __fastcall sub_2EE2EE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rbx
  __m128i v40; // xmm4
  __m128i v41; // xmm3
  __m128i v42; // xmm2
  __m128i v43; // xmm1
  __m128i v44; // xmm0
  char *v45; // rax
  unsigned int v46; // r12d
  __int64 *v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // [rsp+0h] [rbp-560h]
  char v52; // [rsp+Fh] [rbp-551h]
  __int64 v53; // [rsp+10h] [rbp-550h]
  __int64 v54; // [rsp+18h] [rbp-548h]
  __int64 v55; // [rsp+20h] [rbp-540h]
  __int64 v56; // [rsp+28h] [rbp-538h]
  __int64 v57; // [rsp+30h] [rbp-530h]
  __int64 v58; // [rsp+38h] [rbp-528h]
  __int64 v59[11]; // [rsp+40h] [rbp-520h] BYREF
  char v60[320]; // [rsp+98h] [rbp-4C8h] BYREF
  __m128i v61; // [rsp+1D8h] [rbp-388h]
  __m128i v62; // [rsp+1E8h] [rbp-378h]
  __m128i v63; // [rsp+1F8h] [rbp-368h]
  __m128i v64; // [rsp+208h] [rbp-358h]
  __m128i v65; // [rsp+218h] [rbp-348h]
  __m128i v66; // [rsp+228h] [rbp-338h]
  __m128i v67; // [rsp+238h] [rbp-328h]
  __m128i v68; // [rsp+248h] [rbp-318h]
  __m128i v69; // [rsp+258h] [rbp-308h]
  __m128i v70; // [rsp+268h] [rbp-2F8h]
  __int64 v71; // [rsp+278h] [rbp-2E8h]
  __int64 v72; // [rsp+280h] [rbp-2E0h]
  __int64 v73; // [rsp+288h] [rbp-2D8h]
  __int64 v74; // [rsp+290h] [rbp-2D0h]
  __int64 v75; // [rsp+298h] [rbp-2C8h]
  __int64 v76; // [rsp+2A0h] [rbp-2C0h]
  char *v77; // [rsp+2A8h] [rbp-2B8h]
  __int64 v78; // [rsp+2B0h] [rbp-2B0h]
  char v79; // [rsp+2B8h] [rbp-2A8h] BYREF
  __int64 v80; // [rsp+2F8h] [rbp-268h]
  __int64 v81; // [rsp+300h] [rbp-260h]
  __int64 v82; // [rsp+308h] [rbp-258h]
  __int64 v83; // [rsp+310h] [rbp-250h]
  __int64 v84; // [rsp+318h] [rbp-248h]
  char *v85; // [rsp+320h] [rbp-240h]
  __int64 v86; // [rsp+328h] [rbp-238h]
  char v87; // [rsp+330h] [rbp-230h] BYREF
  int v88; // [rsp+3B8h] [rbp-1A8h] BYREF
  __int64 v89; // [rsp+3C0h] [rbp-1A0h]
  int *v90; // [rsp+3C8h] [rbp-198h]
  int *v91; // [rsp+3D0h] [rbp-190h]
  __int64 v92; // [rsp+3D8h] [rbp-188h]
  __int64 v93; // [rsp+3E0h] [rbp-180h]
  __int64 v94; // [rsp+3E8h] [rbp-178h]
  __int64 v95; // [rsp+3F0h] [rbp-170h]
  int v96; // [rsp+3F8h] [rbp-168h]
  __int64 v97; // [rsp+400h] [rbp-160h]
  __int64 v98; // [rsp+408h] [rbp-158h]
  __int64 v99; // [rsp+410h] [rbp-150h]
  __int64 v100; // [rsp+418h] [rbp-148h]
  _QWORD *v101; // [rsp+420h] [rbp-140h]
  __int64 v102; // [rsp+428h] [rbp-138h]
  _QWORD v103[6]; // [rsp+430h] [rbp-130h] BYREF
  char v104; // [rsp+460h] [rbp-100h] BYREF
  _QWORD v105[7]; // [rsp+4A0h] [rbp-C0h] BYREF
  int v106; // [rsp+4D8h] [rbp-88h]
  __int64 v107; // [rsp+4E0h] [rbp-80h]
  __int64 v108; // [rsp+4E8h] [rbp-78h]
  __int64 v109; // [rsp+4F0h] [rbp-70h]
  int v110; // [rsp+4F8h] [rbp-68h]
  __int64 v111; // [rsp+500h] [rbp-60h]
  __int64 v112; // [rsp+508h] [rbp-58h]
  __int64 v113; // [rsp+510h] [rbp-50h]
  int v114; // [rsp+518h] [rbp-48h]
  char v115; // [rsp+520h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_69:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_69;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5027190);
  v6 = *(__int64 **)(a1 + 8);
  v52 = *(_BYTE *)(v5 + 275);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_70:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_501FE44 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_70;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_501FE44);
  v10 = *(__int64 **)(a1 + 8);
  v58 = v9 + 200;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_71:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_50209DC )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_71;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_50209DC);
  v14 = *(__int64 **)(a1 + 8);
  v56 = v13 + 200;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_72:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_501FE3C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_72;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_501FE3C);
  v18 = *(__int64 **)(a1 + 8);
  v55 = v17 + 208;
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
LABEL_73:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F87C64 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_73;
  }
  v54 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
                      *(_QWORD *)(v19 + 8),
                      &unk_4F87C64)
                  + 176);
  if ( (_BYTE)qword_5022228 )
  {
    v48 = *(__int64 **)(a1 + 8);
    v49 = *v48;
    v50 = v48[1];
    if ( v49 == v50 )
LABEL_68:
      BUG();
    while ( *(_UNKNOWN **)v49 != &unk_501EC08 )
    {
      v49 += 16;
      if ( v50 == v49 )
        goto LABEL_68;
    }
    v53 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v49 + 8) + 104LL))(
            *(_QWORD *)(v49 + 8),
            &unk_501EC08)
        + 200;
  }
  else
  {
    v53 = 0;
  }
  v21 = *(__int64 **)(a1 + 8);
  v22 = *v21;
  v23 = v21[1];
  if ( v22 == v23 )
LABEL_74:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_501F1C8 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_74;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_501F1C8);
  v25 = *(__int64 **)(a1 + 8);
  v51 = v24 + 169;
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_75:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_4F86530 )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_75;
  }
  v57 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(
                      *(_QWORD *)(v26 + 8),
                      &unk_4F86530)
                  + 176);
  v28 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501EACC);
  if ( v28 && (v29 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v28 + 104LL))(v28, &unk_501EACC)) != 0 )
    v30 = v29 + 200;
  else
    v30 = 0;
  v31 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5025C1C);
  if ( v31 && (v32 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v31 + 104LL))(v31, &unk_5025C1C)) != 0 )
    v33 = v32 + 200;
  else
    v33 = 0;
  v34 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501EB14);
  if ( v34 && (v35 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v34 + 104LL))(v34, &unk_501EB14)) != 0 )
    v36 = v35 + 200;
  else
    v36 = 0;
  v37 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_50208AC);
  if ( v37 && (v38 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v37 + 104LL))(v37, &unk_50208AC)) != 0 )
    v39 = v38 + 200;
  else
    v39 = 0;
  memset(v59, 0, 32);
  v59[4] = v58;
  v59[5] = v56;
  v59[6] = v55;
  v59[7] = v54;
  v59[8] = v53;
  v59[9] = v51;
  v59[10] = v57;
  sub_2F5FEE0(v60);
  v82 = v33;
  v71 = 0;
  v40 = _mm_loadu_si128(xmmword_3F8F0C0);
  v41 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
  v72 = 0;
  v42 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
  v43 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
  v73 = 0;
  v44 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
  v74 = 0;
  v77 = &v79;
  v78 = 0x1000000000LL;
  v85 = &v87;
  v86 = 0x800000000LL;
  v75 = 0;
  v76 = 0;
  v80 = 0;
  v81 = v30;
  v83 = v36;
  v84 = v39;
  v88 = 0;
  v61 = v40;
  v62 = v41;
  v63 = v42;
  v64 = v43;
  v65 = v44;
  v66 = v40;
  v67 = v41;
  v68 = v42;
  v69 = v43;
  v70 = v44;
  v89 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v102 = 0;
  memset(v103, 0, 40);
  v103[5] = 1;
  v90 = &v88;
  v91 = &v88;
  v101 = v103;
  v45 = &v104;
  do
  {
    *(_DWORD *)v45 = -1;
    v45 += 16;
  }
  while ( v45 != (char *)v105 );
  memset(v105, 0, sizeof(v105));
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = v52;
  v46 = sub_2EE0CF0(v59, a2);
  sub_2ED5940((__int64)v59);
  return v46;
}
