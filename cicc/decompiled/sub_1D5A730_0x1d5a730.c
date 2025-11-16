// Function: sub_1D5A730
// Address: 0x1d5a730
//
_BOOL8 __fastcall sub_1D5A730(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  unsigned __int8 v6; // al

  if ( a2 == a4 || a2 == 0 )
    return 1;
  v5 = a2;
  if ( a3 == a2 )
    return 1;
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 > 0x17u )
  {
    if ( v6 == 53 )
    {
      v5 = a2;
      if ( (unsigned __int8)sub_15F8F00(a2) )
        return 1;
    }
  }
  else if ( v6 != 17 )
  {
    return 1;
  }
  return sub_1648D30(v5, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 40LL));
}
