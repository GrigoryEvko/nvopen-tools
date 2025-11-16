// Function: sub_711520
// Address: 0x711520
//
__int64 __fastcall sub_711520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d

  if ( (unsigned int)sub_72A2A0(a1, a2, a3, a4, a5) )
    return 1;
  v5 = sub_710600(a1);
  if ( v5
    || (*(_QWORD *)(a1 + 168) & 0xFF0000000008LL) == 0x10000000008LL && (unsigned int)sub_8D2660(*(_QWORD *)(a1 + 128)) )
  {
    return 1;
  }
  if ( *(_BYTE *)(a1 + 173) != 7 )
    return v5;
  return sub_737660(a1);
}
