// Function: sub_5C6550
// Address: 0x5c6550
//
__int64 __fastcall sub_5C6550(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  char v5; // al

  *(_BYTE *)(a2 + 201) |= 0x20u;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 40);
    v5 = *(_BYTE *)(v4 + 24);
    if ( v5 != 20 && (v5 != 1 || *(_BYTE *)(v4 + 56) != 5 || *(_BYTE *)(*(_QWORD *)(v4 + 72) + 24LL) != 20) )
      sub_6851C0(3201, v3 + 24);
  }
  return a2;
}
