// Function: sub_1F7DEB0
// Address: 0x1f7deb0
//
__int64 __fastcall sub_1F7DEB0(_QWORD *a1, unsigned int a2, __int64 a3, unsigned int a4, char a5)
{
  unsigned __int8 v7; // dl

  if ( a5 )
  {
    v7 = sub_1D154A0(a2, a4);
    if ( v7 )
      return v7;
  }
  else
  {
    v7 = sub_1D15020(a2, a4);
    if ( v7 )
      return v7;
  }
  return sub_1F593D0(a1, a2, a3, a4);
}
