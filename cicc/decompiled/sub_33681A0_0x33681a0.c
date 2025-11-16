// Function: sub_33681A0
// Address: 0x33681a0
//
_BYTE *__fastcall sub_33681A0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  char v5; // al
  _QWORD v6[4]; // [rsp+0h] [rbp-70h] BYREF
  __int16 v7; // [rsp+20h] [rbp-50h]
  _BYTE v8[64]; // [rsp+30h] [rbp-40h] BYREF

  if ( !a2 || *a2 <= 0x1Cu )
    return sub_B6ECE0(a1, a3);
  if ( *a2 != 85 || **((_BYTE **)a2 - 4) != 25 )
    return sub_B6ED20(a1, (__int64)a2, a3);
  v5 = *(_BYTE *)(a3 + 32);
  if ( v5 )
  {
    if ( v5 == 1 )
    {
      v6[0] = ", possible invalid constraint for vector type";
      v7 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a3 + 33) == 1 )
      {
        a4 = *(_QWORD *)(a3 + 8);
        a3 = *(_QWORD *)a3;
      }
      else
      {
        v5 = 2;
      }
      v6[1] = a4;
      v6[0] = a3;
      v6[2] = ", possible invalid constraint for vector type";
      LOBYTE(v7) = v5;
      HIBYTE(v7) = 3;
    }
  }
  else
  {
    v7 = 256;
  }
  sub_B15700((__int64)v8, (__int64)a2, (__int64)v6, 0);
  return sub_B6EB20(a1, (__int64)v8);
}
