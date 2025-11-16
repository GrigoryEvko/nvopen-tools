// Function: sub_109CDC0
// Address: 0x109cdc0
//
__int64 __fastcall sub_109CDC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // r8d

  v2 = *(_QWORD *)(a2 - 64);
  v3 = *a1;
  v4 = *(_QWORD *)(a2 - 32);
  if ( v2 == *a1 )
  {
    v5 = 1;
    if ( v4 == a1[1] )
      return v5;
    v5 = 0;
    if ( v3 != v4 )
      return v5;
  }
  else
  {
    v5 = 0;
    if ( v3 != v4 )
      return v5;
  }
  LOBYTE(v5) = a1[1] == v2;
  return v5;
}
