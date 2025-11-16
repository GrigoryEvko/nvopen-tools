// Function: sub_2046680
// Address: 0x2046680
//
__int64 __fastcall sub_2046680(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // bl
  unsigned int v6; // r12d
  unsigned int v7; // eax
  _QWORD v8[2]; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int8 v9[8]; // [rsp+10h] [rbp-20h] BYREF
  __int64 v10; // [rsp+18h] [rbp-18h]

  v4 = *(_BYTE *)(a1 + 1160);
  v8[0] = a3;
  v8[1] = a4;
  v9[0] = v4;
  v10 = 0;
  if ( v4 == (_BYTE)a3 )
  {
    if ( v4 || !a4 )
      return v8[0];
LABEL_10:
    v6 = sub_1F58D40((__int64)v8);
    if ( !v4 )
      goto LABEL_11;
LABEL_6:
    v7 = sub_2045180(v4);
    goto LABEL_7;
  }
  if ( !(_BYTE)a3 )
    goto LABEL_10;
  v6 = sub_2045180(a3);
  if ( v4 )
    goto LABEL_6;
LABEL_11:
  v7 = sub_1F58D40((__int64)v9);
LABEL_7:
  if ( v7 <= v6 )
    return v8[0];
  return v4;
}
