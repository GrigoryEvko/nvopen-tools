// Function: sub_26995C0
// Address: 0x26995c0
//
__int64 __fastcall sub_26995C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned __int8 v5; // al
  __m128i v6; // xmm0
  _QWORD *v7; // rax
  __int64 v8; // r12
  __int64 (__fastcall *v9)(__int64); // rax
  _BYTE *v10; // rdi
  __int64 (*v11)(void); // rax
  char v12; // al
  __int64 (__fastcall *v13)(__int64); // rax
  _BYTE *v14; // rdi
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 *v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int8 *v24; // r15
  int v25; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rdx
  bool v29; // cc
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  __int64 **v32; // r12
  __int64 *v33; // rdi
  __int64 v34; // rsi
  __int128 v35; // rax
  _QWORD *v36; // rax
  unsigned __int64 v37; // r15
  __int64 v38; // rax
  unsigned __int8 *v39; // r12
  __int64 v40; // rax
  __int16 v41; // ax
  __int64 v42; // rax
  _QWORD *v43; // rax
  _BYTE *v44; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r9
  char v49; // [rsp+Eh] [rbp-5C2h]
  __int64 **v50; // [rsp+10h] [rbp-5C0h]
  __int64 v51; // [rsp+10h] [rbp-5C0h]
  unsigned int v52; // [rsp+18h] [rbp-5B8h]
  __int64 v53; // [rsp+18h] [rbp-5B8h]
  __int64 *v54; // [rsp+18h] [rbp-5B8h]
  __int64 v55; // [rsp+28h] [rbp-5A8h]
  __int64 *v58; // [rsp+48h] [rbp-588h]
  __int64 v59; // [rsp+50h] [rbp-580h] BYREF
  __int64 *v60; // [rsp+58h] [rbp-578h] BYREF
  _BYTE *v61; // [rsp+60h] [rbp-570h] BYREF
  __int64 v62; // [rsp+68h] [rbp-568h]
  _BYTE v63[32]; // [rsp+70h] [rbp-560h] BYREF
  _QWORD v64[10]; // [rsp+90h] [rbp-540h] BYREF
  char v65[344]; // [rsp+E0h] [rbp-4F0h] BYREF
  __int64 v66; // [rsp+238h] [rbp-398h]
  __int64 v67[10]; // [rsp+240h] [rbp-390h] BYREF
  char v68[352]; // [rsp+290h] [rbp-340h] BYREF
  _BYTE v69[24]; // [rsp+3F0h] [rbp-1E0h] BYREF
  __int16 v70; // [rsp+410h] [rbp-1C0h]
  char v71[344]; // [rsp+440h] [rbp-190h] BYREF
  __int64 v72; // [rsp+598h] [rbp-38h]

  v55 = *(_QWORD *)(a2 + 208);
  v2 = *(_QWORD *)(a1 + 72);
  v3 = v2 & 3;
  v4 = v2 & 0xFFFFFFFFFFFFFFFCLL;
  if ( v3 == 3 )
    v4 = *(_QWORD *)(v4 + 24);
  v5 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 )
  {
    if ( v5 == 22 )
    {
      v4 = *(_QWORD *)(v4 + 24);
    }
    else if ( v5 <= 0x1Cu )
    {
      v4 = 0;
    }
    else
    {
      v4 = sub_B43CB0(v4);
    }
  }
  *(_OWORD *)v69 = v4 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v6 = _mm_loadu_si128((const __m128i *)v69);
  *(_QWORD *)v69 = &unk_438A66A;
  *(__m128i *)&v69[8] = v6;
  v7 = sub_25134D0(a2 + 136, (__int64 *)v69);
  v8 = (__int64)v7;
  if ( v7 )
  {
    v8 = v7[3];
    if ( !v8 )
      goto LABEL_66;
    v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 40LL);
    if ( v9 == sub_2505F20 )
      v10 = (_BYTE *)(v8 + 88);
    else
      v10 = (_BYTE *)v9(v8);
    v11 = *(__int64 (**)(void))(*(_QWORD *)v10 + 16LL);
    if ( (char *)v11 == (char *)sub_2505E30 )
      v12 = v10[9];
    else
      v12 = v11();
    if ( v12 )
      sub_250ED80(a2, v8, a1, 1);
    v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 40LL);
    v14 = (_BYTE *)(v13 == sub_2505F20 ? v8 + 88 : v13(v8));
    v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 16LL);
    if ( !(v15 == sub_2505E30 ? v14[9] : ((__int64 (*)(void))v15)()) )
LABEL_66:
      v8 = 0;
  }
  v52 = 1;
  v17 = *(__int64 **)(a1 + 136);
  v58 = &v17[*(unsigned int *)(a1 + 144)];
  if ( v17 == v58 )
    return v52;
  v18 = v8;
  do
  {
    while ( 1 )
    {
      v19 = *v17;
      if ( !v18 || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 112LL))(v18, *v17) )
      {
        v20 = *(_QWORD *)(v19 + 16);
        v21 = v55;
        v22 = 0;
        v61 = v63;
        v62 = 0x400000000LL;
        if ( v20 )
          break;
      }
LABEL_22:
      if ( v58 == ++v17 )
        return v52;
    }
    do
    {
      while ( 1 )
      {
        v24 = *(unsigned __int8 **)(v20 + 24);
        v25 = *v24;
        if ( (unsigned __int8)v25 <= 0x1Cu )
          goto LABEL_30;
        v26 = (unsigned int)(v25 - 34);
        if ( (unsigned __int8)v26 > 0x33u )
          goto LABEL_30;
        v27 = 0x8000000000041LL;
        if ( !_bittest64(&v27, v26) )
          goto LABEL_30;
        v23 = *((_QWORD *)v24 - 4);
        if ( !v23 )
          goto LABEL_29;
        if ( *(_BYTE *)v23 )
          break;
        if ( *(_QWORD *)(v23 + 24) != *((_QWORD *)v24 + 10) )
          v23 = 0;
LABEL_29:
        if ( *(_QWORD *)(v21 + 32592) == v23 )
          goto LABEL_37;
LABEL_30:
        v20 = *(_QWORD *)(v20 + 8);
        if ( !v20 )
          goto LABEL_40;
      }
      if ( *(_QWORD *)(v21 + 32592) )
        goto LABEL_30;
LABEL_37:
      if ( v22 + 1 > (unsigned __int64)HIDWORD(v62) )
      {
        v51 = v21;
        sub_C8D5F0((__int64)&v61, v63, v22 + 1, 8u, v21, v22 + 1);
        v22 = (unsigned int)v62;
        v21 = v51;
      }
      *(_QWORD *)&v61[8 * v22] = v24;
      v22 = (unsigned int)(v62 + 1);
      LODWORD(v62) = v62 + 1;
      v20 = *(_QWORD *)(v20 + 8);
    }
    while ( v20 );
LABEL_40:
    if ( (_DWORD)v22 != 1
      || ((v28 = *(_QWORD *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)),
           v29 = *(_DWORD *)(v28 + 32) <= 0x40u,
           v30 = *(_QWORD **)(v28 + 24),
           v59 = v28,
           v29)
        ? (v31 = v30)
        : (v31 = (_QWORD *)*v30),
          (unsigned __int64)v31 + *(unsigned int *)(a1 + 248) > (unsigned int)qword_4FF45E8) )
    {
      if ( v61 != v63 )
        _libc_free((unsigned __int64)v61);
      goto LABEL_22;
    }
    v32 = (__int64 **)sub_B43CA0(v19);
    v33 = (__int64 *)sub_BCB2B0(*v32);
    if ( *(_DWORD *)(v59 + 32) <= 0x40u )
      v34 = *(_QWORD *)(v59 + 24);
    else
      v34 = **(_QWORD **)(v59 + 24);
    v50 = (__int64 **)sub_BCD420(v33, v34);
    v53 = sub_ACADE0(v50);
    *(_QWORD *)&v35 = sub_BD5D20(v19);
    *(_OWORD *)v69 = v35;
    *(_QWORD *)&v69[16] = "_shared";
    v67[0] = 0x100000003LL;
    v70 = 773;
    v36 = sub_BD2C40(88, unk_3F0FAE8);
    v37 = (unsigned __int64)v36;
    if ( v36 )
      sub_B30000((__int64)v36, (__int64)v32, v50, 0, 7, v53, (__int64)v69, 0, 0, v67[0], 0);
    v38 = sub_BCE3C0(*v32, 0);
    v39 = (unsigned __int8 *)sub_ADAFB0(v37, v38);
    v60 = &v59;
    if ( *(_QWORD *)(a2 + 4392) )
    {
      v40 = sub_B43CB0(v19);
      v54 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 4392))(*(_QWORD *)(a2 + 4400), v40);
      if ( (unsigned __int8)sub_266EEF0(*v54) )
      {
        sub_B174A0((__int64)v67, *(_QWORD *)(a2 + 4408), (__int64)"OMP111", 6, v19);
        sub_267AC90((__int64)v69, &v60, (__int64)v67);
        sub_B18290((__int64)v69, " [", 2u);
        sub_B18290((__int64)v69, "OMP111", 6u);
        sub_B18290((__int64)v69, "]", 1u);
        sub_23FE290((__int64)v64, (__int64)v69, v46, v47, (__int64)v64, v48);
        v66 = v72;
        v64[0] = &unk_49D9D78;
        *(_QWORD *)v69 = &unk_49D9D40;
        sub_23FD590((__int64)v71);
        v67[0] = (__int64)&unk_49D9D40;
        sub_23FD590((__int64)v68);
        sub_1049740(v54, (__int64)v64);
        v64[0] = &unk_49D9D40;
        sub_23FD590((__int64)v65);
      }
    }
    v41 = sub_A74820((_QWORD *)(v19 + 72));
    if ( !HIBYTE(v41) )
    {
      v42 = *(_QWORD *)(v19 - 32);
      if ( v42 && !*(_BYTE *)v42 && *(_QWORD *)(v42 + 24) == *(_QWORD *)(v19 + 80) )
      {
        *(_QWORD *)v69 = *(_QWORD *)(v42 + 120);
        v49 = sub_A74820(v69);
      }
      LOBYTE(v41) = v49;
    }
    sub_B2F770(v37, v41);
    *(_QWORD *)&v69[8] = 0;
    *(_QWORD *)v69 = v19 & 0xFFFFFFFFFFFFFFFCLL | 1;
    nullsub_1518();
    sub_256F570(a2, *(__int64 *)v69, *(__int64 *)&v69[8], v39, 1u);
    sub_2570110(a2, v19);
    sub_2570110(a2, *(_QWORD *)v61);
    v43 = *(_QWORD **)(v59 + 24);
    if ( *(_DWORD *)(v59 + 32) > 0x40u )
      v43 = (_QWORD *)*v43;
    v44 = v61;
    *(_DWORD *)(a1 + 248) += (_DWORD)v43;
    if ( v44 != v63 )
      _libc_free((unsigned __int64)v44);
    v52 = 0;
    ++v17;
  }
  while ( v58 != v17 );
  return v52;
}
