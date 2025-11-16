// Function: sub_37547E0
// Address: 0x37547e0
//
__int64 __fastcall sub_37547E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  bool v6; // zf
  __int64 v7; // r14
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a2 + 48);
  v9[0] = v5;
  if ( v5 )
    sub_B96E90((__int64)v9, v5, 1);
  v6 = *(_BYTE *)(a2 + 62) == 0;
  *(_BYTE *)(a2 + 63) = 1;
  if ( v6 )
  {
    if ( !*(_BYTE *)(a1 + 56) )
    {
      if ( !*(_BYTE *)(a2 + 61) )
      {
LABEL_6:
        v7 = sub_3753880((__int64 *)a1, a2, a3);
        goto LABEL_7;
      }
      goto LABEL_10;
    }
    v7 = sub_3753C50((__int64 *)a1, a2, a3);
    if ( !v7 )
    {
      if ( !*(_BYTE *)(a2 + 61) )
        goto LABEL_6;
LABEL_10:
      v7 = sub_3753600((__int64 *)a1, a2, a3);
    }
  }
  else
  {
    v7 = sub_3753560(a1, a2);
  }
LABEL_7:
  if ( v9[0] )
    sub_B91220((__int64)v9, v9[0]);
  return v7;
}
