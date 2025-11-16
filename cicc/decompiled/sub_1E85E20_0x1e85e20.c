// Function: sub_1E85E20
// Address: 0x1e85e20
//
__int64 __fastcall sub_1E85E20(_BYTE *a1)
{
  char v1; // dl
  char v2; // si
  __int64 result; // rax
  __int64 v4; // rdx

  v1 = *a1 & 2;
  v2 = *a1 & 1;
  result = *(_QWORD *)a1 >> 2;
  if ( (*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
  {
    if ( !v1 )
    {
      if ( v2 )
        return (unsigned __int16)result;
      return result;
    }
  }
  else if ( !v1 )
  {
    LODWORD(v4) = *(_QWORD *)a1 >> 2;
    if ( v2 )
      LODWORD(v4) = (unsigned __int16)result;
    return (unsigned int)v4 * (unsigned __int16)result;
  }
  v4 = *(_QWORD *)a1 >> 18;
  if ( !v2 )
    return (unsigned int)v4 * (unsigned __int16)result;
  return (unsigned __int16)v4 * (unsigned int)(unsigned __int16)result;
}
