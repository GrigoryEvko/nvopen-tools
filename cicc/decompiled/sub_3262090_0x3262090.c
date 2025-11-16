// Function: sub_3262090
// Address: 0x3262090
//
__int64 __fastcall sub_3262090(__int64 a1, unsigned int a2)
{
  __int64 v2; // rsi
  unsigned __int16 v3; // ax
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  char v8; // cl
  unsigned __int16 v9; // [rsp+10h] [rbp-30h] BYREF
  __int64 v10; // [rsp+18h] [rbp-28h]
  __int64 v11; // [rsp+30h] [rbp-10h]
  __int64 v12; // [rsp+38h] [rbp-8h]

  v2 = *(_QWORD *)(a1 + 48) + 16LL * a2;
  v3 = *(_WORD *)v2;
  v4 = *(_QWORD *)(v2 + 8);
  v9 = v3;
  v10 = v4;
  if ( v3 )
  {
    if ( v3 == 1 || (unsigned __int16)(v3 - 504) <= 7u )
      BUG();
    v7 = 16LL * (v3 - 1);
    v8 = byte_444C4A0[v7 + 8];
    result = *(_QWORD *)&byte_444C4A0[v7];
    LOBYTE(v12) = v8;
    v11 = result;
  }
  else
  {
    result = sub_3007260((__int64)&v9);
    v11 = result;
    v12 = v6;
  }
  return result;
}
