// Function: sub_21BD7A0
// Address: 0x21bd7a0
//
__int64 __fastcall sub_21BD7A0(_QWORD *a1)
{
  __int64 result; // rax
  __int64 *v2; // rdx
  __int64 v3; // rdx
  unsigned int v4; // edx

  result = 0;
  if ( (*a1 & 4) == 0 )
  {
    v2 = (__int64 *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v2 )
    {
      v3 = *v2;
      if ( *(_BYTE *)(v3 + 8) == 15 )
      {
        v4 = *(_DWORD *)(v3 + 8) >> 8;
        if ( v4 == 4 )
        {
          return 2;
        }
        else if ( v4 > 4 )
        {
          result = 5;
          if ( v4 != 5 )
            return 4 * (unsigned int)(v4 == 101);
        }
        else
        {
          result = 1;
          if ( v4 != 1 )
            return 3 * (unsigned int)(v4 == 3);
        }
      }
    }
  }
  return result;
}
