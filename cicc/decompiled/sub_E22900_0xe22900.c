// Function: sub_E22900
// Address: 0xe22900
//
__int64 __fastcall sub_E22900(__int64 a1, size_t *a2)
{
  char v3; // al
  int v4; // edx

  ++a2[1];
  if ( (*a2)-- == 1 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  else
  {
    v3 = sub_E20730(a2, 2u, &unk_42B6DB2);
    v4 = 2;
    if ( !v3 )
      v4 = (unsigned __int8)sub_E20730(a2, 1u, "_") != 0;
    return sub_E22720(a1, (__int64 *)a2, v4);
  }
}
