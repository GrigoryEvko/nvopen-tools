// Function: sub_3433840
// Address: 0x3433840
//
unsigned __int64 __fastcall sub_3433840(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // r12d
  __int64 v5; // rsi
  unsigned __int16 v6; // ax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ecx
  unsigned __int16 v11; // [rsp+0h] [rbp-40h] BYREF
  __int64 v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  char v14; // [rsp+18h] [rbp-28h]
  __int64 v15; // [rsp+20h] [rbp-20h]
  __int64 v16; // [rsp+28h] [rbp-18h]

  LOBYTE(a3) = *(_DWORD *)(a1 + 24) == 39 || *(_DWORD *)(a1 + 24) == 15;
  v3 = a3;
  if ( (_BYTE)a3 )
    return v3;
  v5 = *(_QWORD *)(a1 + 48) + 16LL * a2;
  v6 = *(_WORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v11 = v6;
  v12 = v7;
  if ( v6 )
  {
    if ( v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
      BUG();
    v9 = 16LL * (v6 - 1);
    v8 = *(_QWORD *)&byte_444C4A0[v9];
    LOBYTE(v9) = byte_444C4A0[v9 + 8];
  }
  else
  {
    v8 = sub_3007260((__int64)&v11);
    v15 = v8;
    v16 = v9;
  }
  v13 = v8;
  v14 = v9;
  if ( (unsigned __int64)sub_CA1930(&v13) > 0x40 )
    return v3;
  v10 = *(_DWORD *)(a1 + 24);
  if ( v10 > 0x33 )
    return v3;
  else
    return (0x8001800001800uLL >> v10) & 1;
}
