// Function: sub_E27270
// Address: 0xe27270
//
unsigned __int64 __fastcall sub_E27270(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 result; // rax

  v2 = sub_E253C0(a1, a2, 1);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  result = sub_E263F0(a1, a2, v2);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  return result;
}
