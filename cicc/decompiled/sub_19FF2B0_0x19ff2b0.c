// Function: sub_19FF2B0
// Address: 0x19ff2b0
//
__int64 __fastcall sub_19FF2B0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v5; // r15d
  __int64 v6; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 *v12; // r13
  __int64 v13; // rsi
  unsigned __int8 *v14; // rsi
  __int64 v15[2]; // [rsp+0h] [rbp-50h] BYREF
  char v16; // [rsp+10h] [rbp-40h]
  char v17; // [rsp+11h] [rbp-3Fh]

  v5 = *(_DWORD *)(a3 + 8);
  if ( v5 <= 0x40 )
  {
    if ( *(_QWORD *)a3 )
    {
      v6 = (__int64)a2;
      if ( *(_QWORD *)a3 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) )
        return v6;
      goto LABEL_7;
    }
    return 0;
  }
  if ( v5 == (unsigned int)sub_16A57B0(a3) )
    return 0;
  v6 = (__int64)a2;
  if ( v5 == (unsigned int)sub_16A58F0(a3) )
    return v6;
LABEL_7:
  v8 = *a2;
  v17 = 1;
  v16 = 3;
  v15[0] = (__int64)"and.ra";
  v9 = sub_15A1070(v8, a3);
  v10 = sub_15FB440(26, a2, v9, (__int64)v15, a1);
  v11 = *(_QWORD *)(a1 + 48);
  v6 = v10;
  v15[0] = v11;
  if ( !v11 )
  {
    v12 = (__int64 *)(v10 + 48);
    if ( (__int64 *)(v10 + 48) == v15 )
      return v6;
    v13 = *(_QWORD *)(v10 + 48);
    if ( !v13 )
      return v6;
LABEL_12:
    sub_161E7C0((__int64)v12, v13);
    goto LABEL_13;
  }
  v12 = (__int64 *)(v10 + 48);
  sub_1623A60((__int64)v15, v11, 2);
  if ( v12 == v15 )
  {
    if ( v15[0] )
      sub_161E7C0(v6 + 48, v15[0]);
    return v6;
  }
  v13 = *(_QWORD *)(v6 + 48);
  if ( v13 )
    goto LABEL_12;
LABEL_13:
  v14 = (unsigned __int8 *)v15[0];
  *(_QWORD *)(v6 + 48) = v15[0];
  if ( v14 )
    sub_1623210((__int64)v15, v14, (__int64)v12);
  return v6;
}
