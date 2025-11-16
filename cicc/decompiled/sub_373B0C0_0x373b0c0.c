// Function: sub_373B0C0
// Address: 0x373b0c0
//
unsigned __int64 __fastcall sub_373B0C0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  unsigned __int64 v5; // r13

  v5 = sub_373ABD0(a1, a2, *(_BYTE *)(a3 + 24));
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 8) + 25LL) & 4) == 0 && (*(_BYTE *)(sub_321DF00(a2) + 21) & 4) == 0 )
    return v5;
  *a4 = v5;
  return v5;
}
