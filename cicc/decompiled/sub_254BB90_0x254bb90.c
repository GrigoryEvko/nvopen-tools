// Function: sub_254BB90
// Address: 0x254bb90
//
__int64 __fastcall sub_254BB90(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  __int64 v5; // rdx
  unsigned int v6; // r8d

  v4 = *a2;
  v5 = (unsigned int)(v4 - 29);
  if ( (unsigned int)v5 <= 0x21 )
  {
    if ( (unsigned int)v5 <= 0x1F )
    {
      v6 = 0;
      if ( v4 != 31 )
        return v6;
      v6 = 0;
      if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) == 1 )
        return v6;
    }
    return (unsigned int)sub_B19060(a1 + 200, (__int64)a2, v5, a4) ^ 1;
  }
  if ( (unsigned int)(v4 - 65) <= 1 )
    return (unsigned int)sub_B19060(a1 + 200, (__int64)a2, v5, a4) ^ 1;
  return 0;
}
