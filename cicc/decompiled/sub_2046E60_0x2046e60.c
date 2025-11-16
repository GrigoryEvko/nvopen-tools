// Function: sub_2046E60
// Address: 0x2046e60
//
__int64 __fastcall sub_2046E60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  unsigned __int8 v4; // al
  char v6; // al
  _QWORD v7[2]; // [rsp-28h] [rbp-28h] BYREF
  __int16 v8; // [rsp-18h] [rbp-18h]
  __int64 v9; // [rsp-8h] [rbp-8h]

  if ( !a2 )
    return sub_1602AC0(a1, a3);
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 <= 0x17u )
    BUG();
  if ( v4 != 78 || *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) != 20 )
    return sub_1602B00(a1, a2, a3);
  v9 = v3;
  v6 = *(_BYTE *)(a3 + 16);
  if ( v6 )
  {
    if ( v6 == 1 )
    {
      v7[0] = ", possible invalid constraint for vector type";
      v8 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a3 + 17) == 1 )
        a3 = *(_QWORD *)a3;
      else
        v6 = 2;
      v7[0] = a3;
      v7[1] = ", possible invalid constraint for vector type";
      LOBYTE(v8) = v6;
      HIBYTE(v8) = 3;
    }
  }
  else
  {
    v8 = 256;
  }
  return sub_1602B00(a1, a2, (__int64)v7);
}
