// Function: sub_12FABB0
// Address: 0x12fabb0
//
__int64 __fastcall sub_12FABB0(unsigned __int64 a1, unsigned __int64 a2)
{
  __int64 v4; // r9
  unsigned __int64 v5; // rbx
  __int64 v6; // rsi
  unsigned __int64 v7; // rdx
  __int64 result; // rax
  _QWORD v9[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = HIWORD(a2) & 0x7FFF;
  v5 = a2 >> 63;
  v6 = a2 & 0xFFFFFFFFFFFFLL;
  if ( v4 != 0x7FFF )
  {
    v7 = a1;
    if ( !v4 )
    {
      result = v6 | a1;
      if ( !(v6 | a1) )
        return result;
      sub_12FC3F0(v9, v6, a1);
      v4 = v9[0];
      v6 = v9[2];
      v7 = v9[1];
    }
    return sub_12FBEC0((unsigned __int8)v5, v4, (v7 >> 49) | (v6 << 15) | 0x8000000000000000LL, v7 << 15, 80);
  }
  result = 0x8000000000000000LL;
  if ( a1 | v6 )
  {
    sub_12FBA70(a2, a1, v9);
    return sub_12FBA40(v9);
  }
  return result;
}
