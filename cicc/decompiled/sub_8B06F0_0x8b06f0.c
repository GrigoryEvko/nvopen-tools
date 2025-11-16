// Function: sub_8B06F0
// Address: 0x8b06f0
//
__m128i *__fastcall sub_8B06F0(__int64 a1, int *a2, unsigned int a3)
{
  int v6; // eax
  __m128i v7; // xmm0
  __int64 v8; // rdi
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  _DWORD *v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned int v25; // r15d
  unsigned __int16 v26; // ax
  char v27; // r8
  char v28; // di
  char v29; // si
  int v30; // ecx
  int v31; // edx
  __int64 *v32; // rax
  __m128i *v33; // r8
  __int64 v34; // r12
  __int64 v35; // rax
  unsigned int v36; // r8d
  __m128i *v37; // r14
  unsigned __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned __int64 v44; // rdi
  __int64 *v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r9
  unsigned int *v49; // rsi
  __int64 v50; // r8
  __int64 *v51; // rax
  __int64 v52; // rcx
  int v53; // eax
  int v54; // edx
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  unsigned int v59; // [rsp+0h] [rbp-4A0h]
  __int64 v60; // [rsp+0h] [rbp-4A0h]
  __m128i *v61; // [rsp+0h] [rbp-4A0h]
  __m128i *v62; // [rsp+0h] [rbp-4A0h]
  __int64 v63; // [rsp+8h] [rbp-498h]
  __m128i v64[2]; // [rsp+10h] [rbp-490h] BYREF
  __m128i v65[6]; // [rsp+30h] [rbp-470h] BYREF
  _QWORD v66[60]; // [rsp+90h] [rbp-410h] BYREF
  _BYTE v67[44]; // [rsp+270h] [rbp-230h] BYREF
  int v68; // [rsp+29Ch] [rbp-204h]
  int v69; // [rsp+2BCh] [rbp-1E4h]
  int v70; // [rsp+2C0h] [rbp-1E0h]
  int v71; // [rsp+2C4h] [rbp-1DCh]
  int v72; // [rsp+2C8h] [rbp-1D8h]
  int v73; // [rsp+2F0h] [rbp-1B0h]
  int v74; // [rsp+2F4h] [rbp-1ACh]
  unsigned int v75; // [rsp+2F8h] [rbp-1A8h]
  int v76; // [rsp+318h] [rbp-188h]
  __int64 v77; // [rsp+330h] [rbp-170h]
  __int64 v78; // [rsp+338h] [rbp-168h]
  __int64 v79; // [rsp+348h] [rbp-158h]
  __int64 v80; // [rsp+358h] [rbp-148h]
  __m128i v81; // [rsp+368h] [rbp-138h]
  __m128i v82; // [rsp+378h] [rbp-128h]
  unsigned __int64 v83; // [rsp+458h] [rbp-48h]
  int *v84; // [rsp+460h] [rbp-40h]

  memset(v65, 0, 0x58u);
  memset(v66, 0, 0x1D8u);
  v66[19] = v66;
  v65[2].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
  v66[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v66[22]) |= 1u;
  sub_891F00((__int64)v67, (__int64)v66);
  v6 = *(_DWORD *)(a1 + 44);
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 248));
  v8 = (__int64)v67;
  v9 = _mm_loadu_si128((const __m128i *)(a1 + 264));
  v76 = 0;
  v68 = v6;
  v10 = *(_QWORD *)(a1 + 192);
  v69 = 1;
  v77 = v10;
  v11 = *(_QWORD *)(a1 + 200);
  v84 = a2;
  v78 = v11;
  v12 = *(_QWORD *)(a1 + 232);
  v81 = v7;
  v80 = v12;
  LODWORD(v12) = *(_DWORD *)(a1 + 80);
  v82 = v9;
  v70 = v12;
  sub_8B0460((unsigned __int64)v67, 0, 1, v13, v14, v15);
  for ( ; v79; --v79 )
  {
    v8 = 0x800000;
    sub_862F90(0x800000, 0, v16, v17, v18, v19);
  }
  if ( !v77 )
  {
    v77 = sub_878CA0();
    *(_QWORD *)(v77 + 16) = v80;
  }
  v20 = &dword_4D04964;
  v21 = (unsigned int)dword_4D04964;
  v65[2].m128i_i64[1] = qword_4F063F0;
  if ( dword_4D04964 && ((v20 = &dword_4F077C4, dword_4F077C4 != 2) || (v20 = &unk_4F07778, unk_4F07778 <= 201401)) )
  {
    if ( word_4F06418[0] != 151 )
    {
      if ( word_4F06418[0] != 101 )
      {
        if ( word_4F06418[0] == 183 )
        {
          v21 = 1554;
          v8 = unk_4F07471;
          sub_684AA0(unk_4F07471, 0x612u, &dword_4F063F8);
          goto LABEL_19;
        }
        v8 = 995;
        goto LABEL_14;
      }
LABEL_52:
      v21 = (__int64)dword_4F07508;
      v8 = 996;
      sub_6851C0(0x3E4u, dword_4F07508);
    }
  }
  else if ( word_4F06418[0] != 151 )
  {
    if ( word_4F06418[0] != 101 )
    {
      v8 = 2653;
      if ( word_4F06418[0] == 183 )
        goto LABEL_19;
LABEL_14:
      v21 = (__int64)&dword_4F063F8;
      sub_6851C0(v8, &dword_4F063F8);
      goto LABEL_20;
    }
    goto LABEL_52;
  }
LABEL_19:
  sub_7B8B50(v8, (unsigned int *)v21, (__int64)v20, v17, v18, (__int64)v19);
LABEL_20:
  v25 = 0;
  v26 = word_4F06418[0];
  if ( word_4F06418[0] == 76 )
  {
    v25 = dword_4D04408;
    if ( !dword_4D04408 )
    {
      v27 = v73;
      v28 = v72;
      v29 = v71;
      v30 = *a2;
      v31 = *(_DWORD *)(a1 + 168);
      goto LABEL_22;
    }
    v52 = v75;
    if ( v75 && sub_867060(v8, v21, v22, v75) )
    {
      sub_867610(v8, (unsigned int *)v21);
      v25 = 1;
      v26 = word_4F06418[0];
    }
    else
    {
      sub_7B8B50(v8, (unsigned int *)v21, (__int64)v22, v52, v23, v24);
      v25 = 1;
      v26 = word_4F06418[0];
    }
  }
  v27 = v73;
  v28 = v72;
  v29 = v71;
  v30 = *a2;
  v31 = *(_DWORD *)(a1 + 168);
  if ( v26 != 1 )
  {
LABEL_22:
    v32 = sub_898B40(v83, 0, v31, v30, 0, a3, v25, v29, v28, v27);
    v33 = (__m128i *)v32[11];
    v34 = (__int64)v32;
    v63 = v33[6].m128i_i64[1];
    goto LABEL_23;
  }
  v44 = v83;
  v45 = sub_898B40(v83, (__int64)&qword_4D04A00, v31, v30, 1, a3, v25, v71, v72, v73);
  v49 = (unsigned int *)&qword_4D04A00;
  v50 = v45[11];
  v34 = (__int64)v45;
  v63 = *(_QWORD *)(v50 + 104);
  v51 = *(__int64 **)v77;
  if ( *(_QWORD *)v77 )
  {
    v47 = qword_4D04A00;
    while ( 1 )
    {
      v46 = (_QWORD *)v51[1];
      if ( *v46 == qword_4D04A00 )
        break;
      v51 = (__int64 *)*v51;
      if ( !v51 )
        goto LABEL_34;
    }
    v49 = (unsigned int *)&qword_4D04A08;
    v44 = 1006;
    v60 = v50;
    sub_6851C0(0x3EEu, &qword_4D04A08);
    v50 = v60;
  }
LABEL_34:
  v61 = (__m128i *)v50;
  v65[1].m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
  v65[1].m128i_i64[1] = qword_4F063F0;
  sub_7B8B50(v44, v49, (__int64)v46, v47, v50, v48);
  v33 = v61;
LABEL_23:
  if ( dword_4F07590 )
  {
    v35 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v35 + 4) == 8 )
    {
      if ( *(_QWORD *)(v35 + 184) )
      {
        v62 = v33;
        sub_72EE40(v63, 0x3Bu, *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 184));
        sub_7344C0(v63, dword_4F04C64);
        v33 = v62;
      }
    }
  }
  sub_879080(v33, 0, v77);
  v36 = dword_4F04C3C;
  dword_4F04C3C = 1;
  v59 = v36;
  sub_8756F0(3, v34, (_QWORD *)(v34 + 48), 0);
  dword_4F04C3C = v59;
  sub_729470(v63, v65);
  v37 = (__m128i *)sub_880AD0(v34);
  v38 = *(_QWORD *)v77;
  sub_88F9D0(*(__int64 **)v77, 0);
  if ( v25 )
  {
    v53 = sub_866580();
    v54 = v74;
    if ( v74 && v53 || (v37[3].m128i_i8[8] |= 0x10u, v54) )
    {
      *(_BYTE *)(v34 + 84) |= 0x20u;
      v37[3].m128i_i8[8] |= 0x40u;
    }
    *(_QWORD *)(a1 + 84) = 0x100000001LL;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 1u;
    v37[3].m128i_i8[8] = (32 * (v75 & 1)) | v37[3].m128i_i8[8] & 0xDF;
    *(_BYTE *)(v34 + 84) = ((v75 & 1) << 6) | *(_BYTE *)(v34 + 84) & 0xBF;
    if ( word_4F06418[0] == 56 )
    {
      sub_6851C0(0x77Au, &dword_4F063F8);
      v37[3].m128i_i8[8] &= ~1u;
      sub_7B8B50(0x77Au, &dword_4F063F8, v55, v56, v57, v58);
      sub_6790F0((__int64)v64, 1, 0, 0, 0);
      v37[3].m128i_i8[9] &= ~1u;
      sub_7AEA70(v64);
    }
LABEL_27:
    if ( a3 )
      goto LABEL_28;
LABEL_39:
    sub_879080(v37 + 1, 0, *(_QWORD *)(a1 + 192));
    goto LABEL_28;
  }
  if ( word_4F06418[0] != 56 )
    goto LABEL_27;
  v37[3].m128i_i8[8] |= 1u;
  sub_7B8B50(v38, 0, v39, v40, v41, v42);
  sub_6790F0((__int64)v64, 1, 0, 0, 0);
  v37[3].m128i_i8[9] = (a3 ^ 1) & 1 | v37[3].m128i_i8[9] & 0xFE;
  if ( a3 || *(_DWORD *)(a1 + 76) )
  {
    sub_7BC270(v64);
    sub_88EF30((__int64)v37);
  }
  sub_879080(v37 + 6, v64, *(_QWORD *)(a1 + 192));
  if ( !a3 )
    goto LABEL_39;
LABEL_28:
  *(_BYTE *)(v34 + 83) &= ~0x40u;
  return v37;
}
