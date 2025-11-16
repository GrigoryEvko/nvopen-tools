// Function: sub_18A3AD0
// Address: 0x18a3ad0
//
__int64 __fastcall sub_18A3AD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int i; // r15d
  __int64 j; // r14

  v2 = *(_QWORD *)(a1 + 104);
  for ( i = *(_DWORD *)(a1 + 72); a1 + 88 != v2; v2 = sub_220EF30(v2) )
  {
    for ( j = *(_QWORD *)(v2 + 64); v2 + 48 != j; j = sub_220EF30(j) )
    {
      if ( sub_1441CD0(a2, *(_QWORD *)(j + 80)) )
        i += sub_18A3AD0(j + 64, a2);
    }
  }
  return i;
}
