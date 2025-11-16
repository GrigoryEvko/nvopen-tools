// Function: sub_16991E0
// Address: 0x16991e0
//
__int64 __fastcall sub_16991E0(__int64 a1, __int64 a2, char a3)
{
  char v4; // al
  int v5; // esi
  signed int v6; // r14d
  unsigned int v7; // r14d
  _BOOL8 v9; // rdx
  _QWORD v10[8]; // [rsp+0h] [rbp-40h] BYREF

  v4 = *(_BYTE *)(a2 + 18) ^ *(_BYTE *)(a1 + 18);
  v5 = *(__int16 *)(a2 + 16);
  v6 = *(__int16 *)(a1 + 16) - v5;
  if ( ((v4 & 8) != 0) != a3 )
  {
    sub_16986C0(v10, (__int64 *)a2);
    if ( v6 )
    {
      if ( v6 > 0 )
      {
        v7 = sub_1698C00((__int64)v10, v6 - 1);
        sub_1698CA0(a1, 1);
        sub_1698920(a1, (__int64)v10, v7 != 0);
        if ( v7 != 1 )
          goto LABEL_10;
        goto LABEL_15;
      }
      v7 = sub_1698C00(a1, ~v6);
      sub_1698CA0((__int64)v10, 1);
      v9 = v7 != 0;
    }
    else
    {
      v7 = sub_1698CF0(a1, (__int64)v10);
      if ( v7 )
      {
        v7 = 0;
        sub_1698920(a1, (__int64)v10, 0);
        goto LABEL_4;
      }
      v9 = 0;
    }
    sub_1698920((__int64)v10, a1, v9);
    sub_16985E0(a1, (__int64)v10);
    *(_BYTE *)(a1 + 18) = ~*(_BYTE *)(a1 + 18) & 8 | *(_BYTE *)(a1 + 18) & 0xF7;
    if ( v7 != 1 )
    {
LABEL_10:
      if ( v7 == 3 )
        v7 = 1;
      goto LABEL_4;
    }
LABEL_15:
    v7 = 3;
    goto LABEL_4;
  }
  if ( v6 <= 0 )
  {
    v7 = sub_1698C00(a1, v5 - *(__int16 *)(a1 + 16));
    sub_16988D0(a1, a2);
    return v7;
  }
  sub_16986C0(v10, (__int64 *)a2);
  v7 = sub_1698C00((__int64)v10, v6);
  sub_16988D0(a1, (__int64)v10);
LABEL_4:
  sub_1698460((__int64)v10);
  return v7;
}
