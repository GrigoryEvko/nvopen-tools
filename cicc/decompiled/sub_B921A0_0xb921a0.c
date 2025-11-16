// Function: sub_B921A0
// Address: 0xb921a0
//
__int64 __fastcall sub_B921A0(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v3; // rdi

  if ( *(_BYTE *)a1 == 22 )
  {
    v1 = *(_QWORD *)(a1 + 24);
    if ( v1 )
      return sub_B92180(v1);
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 40);
    if ( v3 )
    {
      v1 = *(_QWORD *)(v3 + 72);
      if ( v1 )
        return sub_B92180(v1);
    }
  }
  return 0;
}
