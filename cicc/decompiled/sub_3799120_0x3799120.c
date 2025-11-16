// Function: sub_3799120
// Address: 0x3799120
//
__int64 __fastcall sub_3799120(__int64 a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int16 v5; // cx
  __int64 v6; // r9
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r15
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v16; // rsi
  __int64 v17; // r15
  unsigned __int8 *v18; // rax
  __int64 v19; // rsi
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // r15
  unsigned __int8 *v23; // rax
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int16 v25; // [rsp+14h] [rbp-5Ch]
  unsigned int v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  int v29; // [rsp+38h] [rbp-38h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(unsigned __int64 **)(a2 + 40);
  LOWORD(v26) = v5;
  v8 = *v7;
  v9 = v7[1];
  v25 = v5;
  v27 = v6;
  v24 = v6;
  v10 = sub_37946F0(a1, v8, v9);
  v11 = a1;
  v12 = v10;
  v14 = *(_QWORD *)(v10 + 48) + 16LL * v13;
  if ( *(_WORD *)v14 == v25 )
  {
    if ( *(_QWORD *)(v14 + 8) == v24 || v25 )
      return v12;
LABEL_13:
    v20 = sub_3007030((__int64)&v26);
    v11 = a1;
    if ( !v20 )
      goto LABEL_14;
LABEL_9:
    v16 = *(_QWORD *)(a2 + 80);
    v17 = *(_QWORD *)(v11 + 8);
    v28 = v16;
    if ( v16 )
      sub_B96E90((__int64)&v28, v16, 1);
    v29 = *(_DWORD *)(a2 + 72);
    v18 = sub_33FAF80(v17, 233, (__int64)&v28, v26, v27, (unsigned int)&v28, a3);
    v19 = v28;
    v12 = (__int64)v18;
    if ( v28 )
      goto LABEL_12;
    return v12;
  }
  if ( !v25 )
    goto LABEL_13;
  if ( (unsigned __int16)(v25 - 10) <= 6u
    || (unsigned __int16)(v25 - 126) <= 0x31u
    || (unsigned __int16)(v25 - 208) <= 0x14u )
  {
    goto LABEL_9;
  }
LABEL_14:
  v21 = *(_QWORD *)(a2 + 80);
  v22 = *(_QWORD *)(v11 + 8);
  v28 = v21;
  if ( v21 )
    sub_B96E90((__int64)&v28, v21, 1);
  v29 = *(_DWORD *)(a2 + 72);
  v23 = sub_33FAF80(v22, 215, (__int64)&v28, v26, v27, (unsigned int)&v28, a3);
  v19 = v28;
  v12 = (__int64)v23;
  if ( v28 )
LABEL_12:
    sub_B91220((__int64)&v28, v19);
  return v12;
}
