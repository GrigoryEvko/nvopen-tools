// Function: sub_20CADE0
// Address: 0x20cade0
//
__int64 __fastcall sub_20CADE0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r14d
  __int64 (*v11)(); // rax
  int v14; // eax
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rbx
  int v23; // r14d
  __int64 **v24; // r15
  __int64 v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rax
  double v29; // xmm4_8
  double v30; // xmm5_8
  __int64 v31; // rdi
  __int64 v32; // r14
  void (*v33)(); // rax
  int v34; // r9d
  _QWORD *v35; // rax
  _QWORD *v36; // r13
  unsigned __int64 *v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  double v43; // xmm4_8
  double v44; // xmm5_8
  unsigned int v45; // [rsp+Ch] [rbp-B4h]
  unsigned __int8 *v46; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int8 *v47[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v48; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v49; // [rsp+40h] [rbp-80h] BYREF
  __int64 v50; // [rsp+48h] [rbp-78h]
  unsigned __int64 *v51; // [rsp+50h] [rbp-70h]
  __int64 v52; // [rsp+58h] [rbp-68h]
  __int64 v53; // [rsp+60h] [rbp-60h]
  int v54; // [rsp+68h] [rbp-58h]
  __int64 v55; // [rsp+70h] [rbp-50h]
  __int64 v56; // [rsp+78h] [rbp-48h]

  v10 = 0;
  v11 = *(__int64 (**)())(**(_QWORD **)(a1 + 160) + 664LL);
  if ( v11 == sub_1F3CB10 )
    return v10;
  v14 = v11();
  if ( v14 == 2 )
  {
    v25 = sub_16498A0(a2);
    v26 = *(unsigned __int8 **)(a2 + 48);
    v49 = 0;
    v52 = v25;
    v27 = *(_QWORD *)(a2 + 40);
    v53 = 0;
    v50 = v27;
    v54 = 0;
    v55 = 0;
    v56 = 0;
    v51 = (unsigned __int64 *)(a2 + 24);
    v47[0] = v26;
    if ( v26 )
    {
      sub_1623A60((__int64)v47, (__int64)v26, 2);
      if ( v49 )
        sub_161E7C0((__int64)&v49, (__int64)v49);
      v49 = v47[0];
      if ( v47[0] )
        sub_1623210((__int64)v47, v47[0], (__int64)&v49);
    }
    v28 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 160) + 608LL))(
            *(_QWORD *)(a1 + 160),
            &v49,
            *(_QWORD *)(a2 - 24),
            (*(unsigned __int16 *)(a2 + 18) >> 7) & 7);
    v31 = *(_QWORD *)(a1 + 160);
    v32 = v28;
    v33 = *(void (**)())(*(_QWORD *)v31 + 640LL);
    if ( v33 != nullsub_760 )
      ((void (__fastcall *)(__int64, unsigned __int8 **))v33)(v31, &v49);
    sub_164D160(a2, v32, a3, a4, a5, a6, v29, v30, a9, a10);
    sub_15F20C0((_QWORD *)a2);
    if ( v49 )
      sub_161E7C0((__int64)&v49, (__int64)v49);
    return 1;
  }
  if ( v14 > 2 )
  {
    v17 = sub_16498A0(a2);
    v20 = *(unsigned __int8 **)(a2 + 48);
    v49 = 0;
    v52 = v17;
    v21 = *(_QWORD *)(a2 + 40);
    v53 = 0;
    v50 = v21;
    v54 = 0;
    v55 = 0;
    v56 = 0;
    v51 = (unsigned __int64 *)(a2 + 24);
    v47[0] = v20;
    if ( v20 )
    {
      sub_1623A60((__int64)v47, (__int64)v20, 2);
      if ( v49 )
        sub_161E7C0((__int64)&v49, (__int64)v49);
      v20 = v47[0];
      v49 = v47[0];
      if ( v47[0] )
        sub_1623210((__int64)v47, v47[0], (__int64)&v49);
    }
    v22 = *(_QWORD *)(a2 - 24);
    v23 = (*(unsigned __int16 *)(a2 + 18) >> 7) & 7;
    v24 = (__int64 **)sub_15A06D0(*(__int64 ***)(*(_QWORD *)v22 + 24LL), (__int64)v20, v18, v19);
    switch ( v23 )
    {
      case 0:
      case 1:
      case 3:
        BUG();
      case 2:
      case 5:
        v34 = 2;
        break;
      case 4:
      case 6:
        v34 = 4;
        break;
      case 7:
        v34 = 7;
        break;
    }
    v45 = v34;
    v48 = 257;
    v35 = sub_1648A60(64, 3u);
    v36 = v35;
    if ( v35 )
      sub_15F99E0((__int64)v35, v22, v24, (__int64)v24, v23, v45, 1, 0);
    if ( v50 )
    {
      v37 = v51;
      sub_157E9D0(v50 + 40, (__int64)v36);
      v38 = v36[3];
      v39 = *v37;
      v36[4] = v37;
      v39 &= 0xFFFFFFFFFFFFFFF8LL;
      v36[3] = v39 | v38 & 7;
      *(_QWORD *)(v39 + 8) = v36 + 3;
      *v37 = *v37 & 7 | (unsigned __int64)(v36 + 3);
    }
    sub_164B780((__int64)v36, (__int64 *)v47);
    if ( v49 )
    {
      v46 = v49;
      sub_1623A60((__int64)&v46, (__int64)v49, 2);
      v40 = v36[6];
      if ( v40 )
        sub_161E7C0((__int64)(v36 + 6), v40);
      v41 = v46;
      v36[6] = v46;
      if ( v41 )
        sub_1623210((__int64)&v46, v41, (__int64)(v36 + 6));
    }
    v47[0] = (unsigned __int8 *)"loaded";
    v48 = 259;
    LODWORD(v46) = 0;
    v42 = sub_12A9E60((__int64 *)&v49, (__int64)v36, (__int64)&v46, 1, (__int64)v47);
    sub_164D160(a2, v42, a3, a4, a5, a6, v43, v44, a9, a10);
    sub_15F20C0((_QWORD *)a2);
    if ( v49 )
    {
      v10 = 1;
      sub_161E7C0((__int64)&v49, (__int64)v49);
      return v10;
    }
    return 1;
  }
  if ( v14 )
  {
    v10 = 1;
    sub_20C96A0(
      a1,
      (_QWORD *)a2,
      *(_QWORD *)(a2 - 24),
      (*(unsigned __int16 *)(a2 + 18) >> 7) & 7,
      (__int64 (__fastcall *)(__int64, unsigned __int8 **, __int64))sub_20C9600,
      (__int64)&v49,
      a3,
      a4,
      a5,
      a6,
      v15,
      v16,
      a9,
      a10);
  }
  return v10;
}
