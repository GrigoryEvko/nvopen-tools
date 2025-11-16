// Function: sub_B13880
// Address: 0xb13880
//
__int64 __fastcall sub_B13880(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  unsigned __int8 **v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  _QWORD *v11; // r13
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r14
  _QWORD *v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-A8h]
  __int64 v39; // [rsp+8h] [rbp-A8h]
  __int64 v40; // [rsp+10h] [rbp-A0h]
  __int64 v41; // [rsp+10h] [rbp-A0h]
  _QWORD v43[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v44; // [rsp+40h] [rbp-70h]
  _QWORD v45[4]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v46; // [rsp+70h] [rbp-40h]
  __int64 v47; // [rsp+78h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 24);
  v45[0] = v5;
  if ( v5 )
    sub_B96E90(v45, v5, 1);
  v6 = sub_B10CD0((__int64)v45);
  v7 = *(_BYTE *)(v6 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(unsigned __int8 ***)(v6 - 32);
  else
    v8 = (unsigned __int8 **)(v6 - 16 - 8LL * ((v7 >> 2) & 0xF));
  sub_AF34D0(*v8);
  if ( v45[0] )
    sub_B91220(v45);
  v9 = *(_QWORD *)(a1 + 24);
  v45[0] = v9;
  if ( v9 )
    sub_B96E90(v45, v9, 1);
  v10 = *(_QWORD *)(sub_B10CD0((__int64)v45) + 8);
  v11 = (_QWORD *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v10 & 4) != 0 )
    v11 = (_QWORD *)*v11;
  if ( v45[0] )
    sub_B91220(v45);
  v12 = *(_BYTE *)(a1 + 64);
  if ( v12 == 1 )
  {
    v37 = sub_B6E160(a2, 71, 0, 0);
    v14 = *(_QWORD *)(a1 + 40);
    v15 = a1 + 80;
    v16 = a1 + 72;
    v3 = v37;
    if ( *(_BYTE *)(a1 + 64) != 2 )
      goto LABEL_22;
    goto LABEL_16;
  }
  if ( v12 <= 1u )
  {
    v13 = sub_B6E160(a2, 69, 0, 0);
    v14 = *(_QWORD *)(a1 + 40);
    v15 = a1 + 80;
    v16 = a1 + 72;
    v3 = v13;
    if ( *(_BYTE *)(a1 + 64) == 2 )
      goto LABEL_16;
    goto LABEL_22;
  }
  if ( v12 != 2 )
  {
    if ( (unsigned __int8)(v12 - 3) <= 1u )
      BUG();
    v14 = *(_QWORD *)(a1 + 40);
    v15 = a1 + 80;
    v16 = a1 + 72;
    goto LABEL_22;
  }
  v35 = sub_B6E160(a2, 68, 0, 0);
  v14 = *(_QWORD *)(a1 + 40);
  v15 = a1 + 80;
  v16 = a1 + 72;
  v3 = v35;
  if ( *(_BYTE *)(a1 + 64) == 2 )
  {
LABEL_16:
    v40 = v15;
    v38 = v16;
    v45[0] = sub_B9F6F0(v11, v14);
    v17 = sub_B12000(v38);
    v45[1] = sub_B9F6F0(v11, v17);
    v18 = sub_B11F60(v40);
    v45[2] = sub_B9F6F0(v11, v18);
    v19 = sub_B13870(a1);
    v20 = sub_B9F6F0(v11, v19);
    v21 = *(_BYTE *)(a1 + 64) == 2;
    v45[3] = v20;
    if ( v21 )
      v22 = *(_QWORD *)(a1 + 48);
    else
      v22 = *(_QWORD *)(a1 + 40);
    v46 = sub_B9F6F0(v11, v22);
    v23 = sub_B11F60(a1 + 88);
    v44 = 257;
    v47 = sub_B9F6F0(v11, v23);
    v24 = *(_QWORD *)(v3 + 24);
    v25 = sub_BD2CC0(88, 7);
    if ( v25 )
    {
      sub_B44260(v25, **(_QWORD **)(v24 + 16), 56, 7, 0, 0);
      *(_QWORD *)(v25 + 72) = 0;
      sub_B4A290(v25, v24, v3, (unsigned int)v45, 6, (unsigned int)v43, 0, 0);
    }
    goto LABEL_24;
  }
LABEL_22:
  v41 = v15;
  v39 = v16;
  v43[0] = sub_B9F6F0(v11, v14);
  v26 = sub_B12000(v39);
  v43[1] = sub_B9F6F0(v11, v26);
  v27 = sub_B11F60(v41);
  v43[2] = sub_B9F6F0(v11, v27);
  LOWORD(v46) = 257;
  v28 = *(_QWORD *)(v3 + 24);
  v25 = sub_BD2CC0(88, 4);
  if ( v25 )
  {
    sub_B44260(v25, **(_QWORD **)(v28 + 16), 56, 4, 0, 0);
    *(_QWORD *)(v25 + 72) = 0;
    sub_B4A290(v25, v28, v3, (unsigned int)v43, 3, (unsigned int)v45, 0, 0);
  }
LABEL_24:
  v29 = (_QWORD *)(v25 + 48);
  *(_WORD *)(v25 + 2) = *(_WORD *)(v25 + 2) & 0xFFFC | 1;
  v30 = *(_QWORD *)(a1 + 24);
  v45[0] = v30;
  if ( !v30 )
  {
    if ( v29 == v45 || !*(_QWORD *)(v25 + 48) )
      goto LABEL_28;
LABEL_35:
    sub_B91220(v25 + 48);
    goto LABEL_36;
  }
  sub_B96E90(v45, v30, 1);
  if ( v29 == v45 )
  {
    if ( v45[0] )
      sub_B91220(v45);
    goto LABEL_28;
  }
  if ( *(_QWORD *)(v25 + 48) )
    goto LABEL_35;
LABEL_36:
  v36 = v45[0];
  *(_QWORD *)(v25 + 48) = v45[0];
  if ( v36 )
    sub_B976B0(v45, v36, v25 + 48, v31, v32, v33);
LABEL_28:
  if ( a3 )
    sub_B44220(v25, a3 + 24, 0);
  return v25;
}
