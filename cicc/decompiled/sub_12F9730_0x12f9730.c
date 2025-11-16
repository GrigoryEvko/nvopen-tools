// Function: sub_12F9730
// Address: 0x12f9730
//
__int64 __fastcall sub_12F9730(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx

  v5 = (unsigned int)a2;
  v6 = ~(unsigned __int64)a1;
  if ( (v6 & 0x7F800000) != 0 || (a1 & 0x7FFFFF) == 0 )
  {
    v6 = ~(unsigned __int64)(unsigned int)a2;
    if ( (~(_DWORD)a2 & 0x7F800000) != 0 || (a2 & 0x7FFFFF) == 0 )
    {
      LODWORD(a5) = 1;
      if ( a1 != (unsigned __int64)(unsigned int)a2 )
        LOBYTE(a5) = 2 * ((unsigned int)a2 | a1) == 0;
      return (unsigned int)a5;
    }
  }
  if ( (a1 & 0x7FC00000) != 0x7F800000 || (a1 & 0x3FFFFF) == 0 )
  {
    a5 = 0;
    if ( (a2 & 0x7FC00000) != 0x7F800000 )
      return (unsigned int)a5;
    a2 &= 0x3FFFFFu;
    if ( !(_DWORD)a2 )
      return (unsigned int)a5;
  }
  sub_12F9B70(16, a2, v5, v6, a5);
  return 0;
}
