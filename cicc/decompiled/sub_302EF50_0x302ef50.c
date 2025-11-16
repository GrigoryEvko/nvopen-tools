// Function: sub_302EF50
// Address: 0x302ef50
//
__int64 __fastcall sub_302EF50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v6; // rdx
  __int64 v7; // rdx
  _QWORD v8[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v9[4]; // [rsp+10h] [rbp-20h] BYREF

  v3 = 0;
  if ( *(_BYTE *)(a2 + 8) == 12 && *(_BYTE *)(a3 + 8) == 12 )
  {
    v8[0] = sub_BCAE30(a2);
    v8[1] = v6;
    if ( sub_CA1930(v8) == 64 )
    {
      v9[0] = sub_BCAE30(a3);
      v9[1] = v7;
      LOBYTE(v3) = sub_CA1930(v9) == 32;
    }
  }
  return v3;
}
