// Function: sub_388F470
// Address: 0x388f470
//
__int64 __fastcall sub_388F470(__int64 a1, _BYTE *a2)
{
  char v3; // [rsp+Bh] [rbp-25h] BYREF
  _DWORD v4[9]; // [rsp+Ch] [rbp-24h] BYREF

  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 310, "expected 'linkage' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  *a2 = sub_3887320(*(_DWORD *)(a1 + 64), &v3) & 0xF | *a2 & 0xF0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 311, "expected 'notEligibleToImport' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_388F090(a1, v4) )
    return 1;
  *a2 = (16 * (v4[0] & 1)) | *a2 & 0xEF;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 312, "expected 'live' here") )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    return 1;
  if ( (unsigned __int8)sub_388F090(a1, v4) )
    return 1;
  *a2 = (32 * (v4[0] & 1)) | *a2 & 0xDF;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388AF10(a1, 313, "expected 'dsoLocal' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388F090(a1, v4) )
  {
    return 1;
  }
  *a2 = ((v4[0] & 1) << 6) | *a2 & 0xBF;
  return sub_388AF10(a1, 13, "expected ')' here");
}
