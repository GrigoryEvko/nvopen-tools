// Function: sub_828690
// Address: 0x828690
//
__int64 __fastcall sub_828690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r12

  LODWORD(v9) = 1;
  if ( !(unsigned int)sub_8B3500(a2, a1, a5, a6, 1) )
  {
    LODWORD(v9) = 0;
    if ( ((unsigned int)sub_8D2E30(a4) || (unsigned int)sub_8D3D10(a4))
      && ((unsigned int)sub_8D2E30(a3) || (unsigned int)sub_8D3D10(a3)) )
    {
      return (unsigned int)sub_8B4FF0(a4, a3, a5, a6, 0) != 0;
    }
  }
  return (unsigned int)v9;
}
