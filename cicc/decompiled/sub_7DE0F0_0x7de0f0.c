// Function: sub_7DE0F0
// Address: 0x7de0f0
//
__int64 __fastcall sub_7DE0F0(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *v3; // r15
  __int64 v4; // rax
  _QWORD *v5; // r14
  unsigned __int8 *v6; // r12
  _QWORD *v7; // r15
  __m128i *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax
  const __m128i *v14; // r13
  __int64 v15; // r12
  __int64 v16; // r13
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r12
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  _BYTE *v24; // rax
  _BYTE *v25; // rax
  const __m128i *v26; // rsi
  _BYTE *v27; // r12
  _QWORD *v28; // rax
  __int64 v29; // r12
  _BYTE *v30; // rax
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  __int64 v34; // rdi
  _BYTE *v35; // r12
  _QWORD *v36; // rax
  __int64 v37; // r12
  _BYTE *v38; // rax
  _BYTE *v39; // rax
  _BYTE *v40; // rax
  _BYTE *v41; // rax
  _QWORD *v42; // rax
  _QWORD *v43; // r12
  __int64 *v44; // r12
  _QWORD *v45; // rax
  __int64 v46; // rsi
  void *v47; // r12
  __int64 v48; // r8
  __int64 *v49; // r15
  __int64 v50; // r12
  _QWORD *v51; // rbx
  __int64 *v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // rbx
  __int64 v57; // rax
  _QWORD *v58; // r12
  __int64 v59; // rax
  __int64 v60; // r15
  __int64 v61; // r15
  __int64 v62; // rdi
  __int64 *v63; // r12
  _QWORD *v64; // rax
  void *v65; // r12
  _QWORD *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rdi
  const __m128i *v70; // r12
  __int64 v71; // rax
  const __m128i *v72; // rsi
  _QWORD *v73; // r12
  _QWORD *v74; // rax
  __int64 v75; // rdi
  __int64 v76; // rdi
  _BYTE *v77; // r12
  _BYTE *v78; // rax
  _BYTE *v79; // rax
  __int64 v81; // r12
  _QWORD *v82; // rdi
  __int64 v83; // r13
  _QWORD *v84; // rax
  __int64 v85; // r13
  __int64 v86; // rax
  __int64 v87; // [rsp+8h] [rbp-148h]
  __int64 v88; // [rsp+8h] [rbp-148h]
  __m128i *v89; // [rsp+8h] [rbp-148h]
  __int64 v90; // [rsp+10h] [rbp-140h]
  __int64 v92; // [rsp+20h] [rbp-130h]
  __int64 v93; // [rsp+20h] [rbp-130h]
  __int64 v94; // [rsp+20h] [rbp-130h]
  __int64 v95; // [rsp+20h] [rbp-130h]
  __int64 v96; // [rsp+20h] [rbp-130h]
  bool v97; // [rsp+20h] [rbp-130h]
  __int64 v98; // [rsp+38h] [rbp-118h]
  __int64 v99; // [rsp+38h] [rbp-118h]
  _BYTE *v100; // [rsp+38h] [rbp-118h]
  __int64 v101; // [rsp+38h] [rbp-118h]
  __int64 v102; // [rsp+38h] [rbp-118h]
  __int64 v103; // [rsp+38h] [rbp-118h]
  void *v104; // [rsp+38h] [rbp-118h]
  __int64 v105; // [rsp+38h] [rbp-118h]
  __int64 v107; // [rsp+40h] [rbp-110h]
  __int64 v108; // [rsp+48h] [rbp-108h]
  __int64 v109; // [rsp+48h] [rbp-108h]
  __int64 v110; // [rsp+50h] [rbp-100h] BYREF
  __int64 v111; // [rsp+58h] [rbp-F8h] BYREF
  _QWORD *v112; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v113; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v114; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v115; // [rsp+78h] [rbp-D8h] BYREF
  _BYTE v116[32]; // [rsp+80h] [rbp-D0h] BYREF
  const __m128i *v117; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v118; // [rsp+A8h] [rbp-A8h]
  _BYTE v119[144]; // [rsp+C0h] [rbp-90h] BYREF

  v3 = *(_QWORD **)(a1 + 72);
  v4 = v3[1];
  v5 = (_QWORD *)v3[2];
  v90 = (__int64)v3;
  v112 = 0;
  v108 = v4;
  sub_7E7090(a1, v116, &v110);
  sub_7DC550(5u, &v111, (__int64)v116);
  v6 = (unsigned __int8 *)v3[3];
  sub_7E18E0(v119, 0, v6);
  v119[25] = a2;
  *(_QWORD *)(unk_4D03F68 + 64LL) = v111;
  if ( unk_4D03F8C )
  {
    sub_733650((__int64)v6);
    sub_732E60(v6, 0x14u, *(_QWORD **)(a1 + 80));
  }
  sub_7E9190(v6, v116);
  v115 = 0;
  v114 = 0;
  v113 = 0;
  if ( v5 )
  {
    v7 = v5;
    do
    {
      v8 = (__m128i *)v7[2];
      if ( v8 )
        v8 = (__m128i *)v8[7].m128i_i64[1];
      sub_7DDD40(v8, &v113, (__int64)&v114, &v115, *v7 == 0);
      v7 = (_QWORD *)*v7;
    }
    while ( v7 );
    v9 = v115;
    v10 = v114;
    v11 = v113;
  }
  else
  {
    v11 = 0;
    v10 = 0;
    v9 = 0;
  }
  v87 = v11;
  v92 = v10;
  v98 = v9;
  v12 = sub_7DC0C0();
  v13 = sub_7DB0A0(v12, 0, &v117);
  v14 = v117;
  v15 = v13;
  v117[11].m128i_i64[0] = v87;
  v14[11].m128i_i64[1] = v92;
  *(_QWORD *)(*(_QWORD *)(v13 + 120) + 176LL) = v98;
  sub_8D6090(*(_QWORD *)(v13 + 120));
  v14[8].m128i_i64[0] = *(_QWORD *)(v15 + 120);
  v16 = v111;
  v117 = (const __m128i *)sub_724DC0();
  v88 = qword_4F18880;
  v93 = qword_4F18890;
  v99 = qword_4F18898;
  v17 = sub_731250(v16);
  v18 = sub_73DE50((__int64)v17, v99);
  v19 = sub_73DE50((__int64)v18, v93);
  v100 = sub_73DE50((__int64)v19, v88);
  v20 = sub_7E2510(v15);
  sub_7E6A80(v100, 73, v20, v116);
  v21 = qword_4F18878;
  v94 = qword_4F18890;
  v101 = qword_4F18898;
  v22 = sub_731250(v16);
  v23 = sub_73DE50((__int64)v22, v101);
  v24 = sub_73DE50((__int64)v23, v94);
  v25 = sub_73DE50((__int64)v24, v21);
  v26 = v117;
  v27 = v25;
  sub_72BB40(*(_QWORD *)(qword_4F18878 + 120), v117);
  v28 = sub_73A720(v117, (__int64)v26);
  sub_7E6A80(v27, 73, v28, v116);
  v29 = qword_4F18870;
  v95 = qword_4F18890;
  v102 = qword_4F18898;
  v30 = sub_731250(v16);
  v31 = sub_73DE50((__int64)v30, v102);
  v32 = sub_73DE50((__int64)v31, v95);
  v33 = sub_73DE50((__int64)v32, v29);
  v34 = qword_4F188E0;
  v35 = v33;
  if ( !qword_4F188E0 )
  {
    sub_72BA30(unk_4F06871);
    qword_4F188E0 = sub_7E2190("__eh_curr_region");
    v34 = qword_4F188E0;
  }
  v36 = sub_73E830(v34);
  sub_7E6A80(v35, 73, v36, v116);
  v37 = qword_4F18888;
  v96 = qword_4F18890;
  v103 = qword_4F18898;
  v38 = sub_731250(v16);
  v39 = sub_73DE50((__int64)v38, v103);
  v40 = sub_73DE50((__int64)v39, v96);
  v41 = sub_73DE50((__int64)v40, v37);
  v42 = (_QWORD *)sub_7E2230(v41);
  v43 = v42;
  if ( qword_4F18828 )
  {
    v44 = (__int64 *)sub_7F88E0(qword_4F18828, v42);
  }
  else
  {
    v83 = *v42;
    v84 = sub_72BA30(5u);
    v44 = (__int64 *)sub_7F8B20(unk_4F06950, &qword_4F18828, v84, v83, 0, v43);
  }
  v45 = sub_73A830(0, 5u);
  v46 = *v44;
  v44[2] = (__int64)v45;
  v47 = sub_73DBF0(0x3Au, v46, (__int64)v44);
  sub_724E30((__int64)&v117);
  v48 = *(_QWORD *)(qword_4F04C50 + 32LL);
  if ( (*(_BYTE *)(v48 + 202) & 1) != 0 && *(_QWORD *)(v48 + 104) )
  {
    v104 = v47;
    v49 = *(__int64 **)(v48 + 104);
    v50 = *(_QWORD *)(qword_4F04C50 + 32LL);
    v51 = 0;
    while ( 1 )
    {
      if ( *((_BYTE *)v49 + 8) != 28 )
        goto LABEL_16;
      *(_BYTE *)(v50 + 202) &= ~1u;
      sub_684B00(0xB3Au, (_DWORD *)v49 + 14);
      v53 = *v49;
      if ( !v51 )
        break;
      *v51 = v53;
      v52 = (__int64 *)*v49;
      v51 = v49;
      if ( !*v49 )
      {
LABEL_21:
        v47 = v104;
        goto LABEL_22;
      }
LABEL_17:
      v49 = v52;
    }
    *(_QWORD *)(v50 + 104) = v53;
LABEL_16:
    v52 = (__int64 *)*v49;
    v51 = v49;
    if ( !*v49 )
      goto LABEL_21;
    goto LABEL_17;
  }
LABEL_22:
  sub_7268E0(v110, 1);
  v54 = v110;
  *(_QWORD *)(v110 + 48) = v47;
  *(_QWORD *)(v54 + 72) = v108;
  v55 = a2;
  sub_7EDF20(v108, a2, 0, a3, &v115);
  v97 = a3 != 0;
  if ( a2 && a3 )
  {
    sub_7E1720(v115, &v117);
    v55 = v108;
    sub_806BE0(&v117, v108, a3);
    if ( (_DWORD)v117 )
      v81 = *(_QWORD *)(v118 + 72);
    else
      v81 = *(_QWORD *)(v118 + 16);
    v82 = qword_4D03F60;
    qword_4D03F60 = (_QWORD *)*qword_4D03F60;
    *v82 = 0;
    sub_7E17F0();
    sub_7F8B60(v81);
  }
  v56 = sub_726B30(7);
  v89 = sub_726410();
  sub_730430((__int64)v89);
  v56[9] = v89;
  v89[8].m128i_i64[0] = (__int64)v56;
  v57 = *(_QWORD *)(v108 + 72);
  v56[3] = v108;
  v56[2] = v57;
  *(_QWORD *)(v108 + 72) = v56;
  sub_7E2B60(v56[2]);
  v109 = v110;
  if ( v5 )
  {
    v107 = 0;
    do
    {
      ++v107;
      v105 = v5[3];
      sub_7EC960();
      v58 = (_QWORD *)unk_4D03F68;
      do
      {
        v58 = (_QWORD *)*v58;
        v59 = v58[1];
        v60 = *(_QWORD *)(v59 + 112);
        if ( v60 )
        {
          do
          {
            if ( (*(_BYTE *)(v60 + 170) & 8) != 0 )
            {
              v55 = (__int64)&v112;
              sub_7DB810(v60, &v112);
            }
            v60 = *(_QWORD *)(v60 + 112);
          }
          while ( v60 );
          v59 = v58[1];
        }
        v61 = *(_QWORD *)(v59 + 120);
        if ( v61 )
        {
          do
          {
            if ( (*(_BYTE *)(v61 + 170) & 8) != 0 )
            {
              v55 = (__int64)&v112;
              sub_7DB810(v61, &v112);
            }
            v61 = *(_QWORD *)(v61 + 112);
          }
          while ( v61 );
          v59 = v58[1];
        }
      }
      while ( qword_4F04C50 != v59 );
      if ( v5[2] || v112 )
      {
        v62 = qword_4F188D0;
        if ( !qword_4F188D0 )
        {
          sub_72BA30(5u);
          qword_4F188D0 = sub_7E2190("__catch_clause_number");
          v62 = qword_4F188D0;
        }
        v63 = sub_73E830(v62);
        v64 = sub_73A830(v107, 5u);
        v55 = *v63;
        v63[2] = (__int64)v64;
        v65 = sub_73DBF0(0x3Au, v55, (__int64)v63);
        v66 = sub_726B30(1);
        *v66 = v5[1];
        v67 = *(_QWORD *)(v105 + 8);
        v66[9] = v105;
        v68 = v109;
        v66[1] = v67;
        v66[6] = v65;
        v109 = (__int64)v66;
        *(_QWORD *)(v68 + 80) = v66;
      }
      else
      {
        *(_QWORD *)(v109 + 80) = v105;
      }
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v5[3] + 80LL) + 8LL) + 32LL) = 0;
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
  }
  if ( v112 )
  {
    v69 = sub_8D46C0(*v112);
    if ( (*(_BYTE *)(v69 + 140) & 0xFB) == 8 )
    {
      v55 = dword_4F077C4 != 2;
      sub_8D4C10(v69, v55);
    }
    v70 = (const __m128i *)sub_724DC0();
    v117 = v70;
    v71 = sub_7E1C10(v69, v55);
    v72 = v70;
    sub_72BB40(v71, v70);
    v73 = sub_73A720(v117, (__int64)v70);
    sub_724E30((__int64)&v117);
    v74 = v112;
    v75 = qword_4F18820;
    v112 = v73;
    v73[2] = v74;
    if ( v75 )
    {
      v76 = sub_7F88E0(v75, v73);
    }
    else
    {
      v85 = sub_7E1C10(0, v72);
      v86 = sub_72CBE0();
      v76 = sub_7F8B20("__suppress_optim_on_vars_in_try", &qword_4F18820, v86, v85, 0, v73);
    }
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(qword_4F18820 + 152) + 168LL) + 16LL) |= 1u;
    v77 = sub_732B10(v76);
    v78 = sub_726B30(11);
    *(_QWORD *)(v109 + 80) = v78;
    sub_7E1740(v78);
    sub_7E6810(v77, &v117, 1);
    v79 = sub_726B30(6);
    *((_QWORD *)v79 + 9) = v89;
    sub_7E6810(v79, &v117, 1);
  }
  if ( !a2 || v97 )
  {
    sub_7E1720(v110, v116);
    sub_7DE060((__int64)v119, v90, (__int64)v116);
  }
  return sub_7E1AA0();
}
