// Function: sub_B13150
// Address: 0xb13150
//
__int64 __fastcall sub_B13150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r12
  _QWORD *v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v18; // rsi
  __int64 v20; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v21[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  v4 = sub_B6E160(a2, 70, 0, 0);
  v5 = sub_B11FB0(a1 + 40);
  v6 = *(_QWORD *)(a1 + 24);
  v7 = v5;
  v21[0] = v6;
  if ( v6 )
    sub_B96E90(v21, v6, 1);
  v8 = *(_QWORD *)(sub_B10CD0((__int64)v21) + 8);
  v9 = (_QWORD *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v8 & 4) != 0 )
    v9 = (_QWORD *)*v9;
  v20 = sub_B9F6F0(v9, v7);
  if ( v21[0] )
    sub_B91220(v21);
  v22 = 257;
  v10 = *(_QWORD *)(v4 + 24);
  v11 = sub_BD2CC0(88, 2);
  if ( v11 )
  {
    sub_B44260(v11, **(_QWORD **)(v10 + 16), 56, 2, 0, 0);
    *(_QWORD *)(v11 + 72) = 0;
    sub_B4A290(v11, v10, v4, (unsigned int)&v20, 1, (unsigned int)v21, 0, 0);
  }
  v12 = (_QWORD *)(v11 + 48);
  *(_WORD *)(v11 + 2) = *(_WORD *)(v11 + 2) & 0xFFFC | 1;
  v13 = *(_QWORD *)(a1 + 24);
  v21[0] = v13;
  if ( !v13 )
  {
    if ( v12 == v21 || !*(_QWORD *)(v11 + 48) )
      goto LABEL_13;
LABEL_17:
    sub_B91220(v11 + 48);
    goto LABEL_18;
  }
  sub_B96E90(v21, v13, 1);
  if ( v12 == v21 )
  {
    if ( v21[0] )
      sub_B91220(v21);
    goto LABEL_13;
  }
  if ( *(_QWORD *)(v11 + 48) )
    goto LABEL_17;
LABEL_18:
  v18 = v21[0];
  *(_QWORD *)(v11 + 48) = v21[0];
  if ( v18 )
    sub_B976B0(v21, v18, v11 + 48, v14, v15, v16);
LABEL_13:
  if ( a3 )
    sub_B44220(v11, a3 + 24, 0);
  return v11;
}
