// Function: sub_B99FD0
// Address: 0xb99fd0
//
void __fastcall sub_B99FD0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rsi
  unsigned __int8 *v4; // rsi
  __int64 v5; // [rsp+8h] [rbp-38h]
  _QWORD v6[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( a3 || *(_QWORD *)(a1 + 48) || (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    if ( a2 )
    {
      if ( a2 == 38 )
      {
        v5 = a3;
        sub_B99BD0(a1, a3);
        a3 = v5;
      }
      sub_B99110(a1, a2, a3);
    }
    else
    {
      sub_B10CB0(v6, a3);
      v3 = *(_QWORD *)(a1 + 48);
      if ( v3 )
        sub_B91220(a1 + 48, v3);
      v4 = (unsigned __int8 *)v6[0];
      *(_QWORD *)(a1 + 48) = v6[0];
      if ( v4 )
        sub_B976B0((__int64)v6, v4, a1 + 48);
    }
  }
}
