// Function: sub_28EA8C0
// Address: 0x28ea8c0
//
__int64 __fastcall sub_28EA8C0(__int64 a1, unsigned __int8 a2, char a3, __int64 a4, __int64 a5)
{
  unsigned int v8; // r14d
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 *v15; // r13
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  __int64 v20[4]; // [rsp+10h] [rbp-60h] BYREF
  char v21; // [rsp+30h] [rbp-40h]
  char v22; // [rsp+31h] [rbp-3Fh]

  v8 = *(_DWORD *)(a5 + 8);
  if ( v8 > 0x40 )
  {
    if ( v8 != (unsigned int)sub_C444A0(a5) )
      goto LABEL_3;
    return 0;
  }
  if ( !*(_QWORD *)a5 )
    return 0;
LABEL_3:
  if ( !v8 )
    return a4;
  if ( v8 > 0x40 )
  {
    if ( v8 != (unsigned int)sub_C445E0(a5) )
      goto LABEL_6;
    return a4;
  }
  if ( *(_QWORD *)a5 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) )
    return a4;
LABEL_6:
  v9 = *(_QWORD *)(a4 + 8);
  v22 = 1;
  v21 = 3;
  v20[0] = (__int64)"and.ra";
  v10 = sub_AD8D80(v9, a5);
  v11 = a2;
  BYTE1(v11) = a3;
  v12 = sub_B504D0(28, a4, v10, (__int64)v20, a1, v11);
  v13 = v12;
  if ( !a1 )
    BUG();
  v14 = *(_QWORD *)(a1 + 24);
  v15 = (__int64 *)(v12 + 48);
  v20[0] = v14;
  if ( !v14 )
  {
    if ( v15 == v20 )
      return v13;
    v17 = *(_QWORD *)(v12 + 48);
    if ( !v17 )
      return v13;
LABEL_18:
    sub_B91220((__int64)v15, v17);
    goto LABEL_19;
  }
  sub_B96E90((__int64)v20, v14, 1);
  if ( v15 == v20 )
  {
    if ( v20[0] )
      sub_B91220((__int64)v15, v20[0]);
    return v13;
  }
  v17 = *(_QWORD *)(v13 + 48);
  if ( v17 )
    goto LABEL_18;
LABEL_19:
  v18 = (unsigned __int8 *)v20[0];
  *(_QWORD *)(v13 + 48) = v20[0];
  if ( v18 )
    sub_B976B0((__int64)v20, v18, (__int64)v15);
  return v13;
}
