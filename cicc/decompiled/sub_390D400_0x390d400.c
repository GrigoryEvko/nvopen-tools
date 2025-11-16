// Function: sub_390D400
// Address: 0x390d400
//
__int64 __fastcall sub_390D400(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  _BYTE *v5; // rsi
  _QWORD v6[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( (*(_BYTE *)(a2 + 44) & 4) != 0 )
    return 0;
  v6[3] = v2;
  v6[0] = a2;
  v5 = *(_BYTE **)(a1 + 40);
  if ( v5 == *(_BYTE **)(a1 + 48) )
  {
    sub_390D270(a1 + 32, v5, v6);
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = a2;
      v5 = *(_BYTE **)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v5 + 8;
  }
  *(_BYTE *)(a2 + 44) |= 4u;
  return 1;
}
