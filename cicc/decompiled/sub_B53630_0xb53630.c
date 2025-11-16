// Function: sub_B53630
// Address: 0xb53630
//
__int64 __fastcall sub_B53630(__int64 a1, __int64 a2)
{
  char v3; // dl
  __int64 v4; // [rsp+34h] [rbp-1Ch]

  if ( (_DWORD)a1 != (_DWORD)a2 )
  {
    if ( (unsigned int)a2 <= 0xF || (unsigned int)a1 <= 0xF )
      return 0;
    if ( BYTE4(a1) && (unsigned int)sub_B53550(a2) == (_DWORD)a1 )
    {
      LODWORD(v4) = a2;
      BYTE4(v4) = 0;
      return v4;
    }
    if ( !BYTE4(a2) || (unsigned int)sub_B53550(a1) != (_DWORD)a2 )
      return 0;
    LODWORD(v4) = a1;
    BYTE4(v4) = 0;
    return v4;
  }
  v3 = BYTE4(a1);
  LODWORD(v4) = a1;
  if ( BYTE4(a1) != BYTE4(a2) )
    v3 = 0;
  BYTE4(v4) = v3;
  return v4;
}
