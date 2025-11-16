// Function: sub_988C10
// Address: 0x988c10
//
char __fastcall sub_988C10(__int64 a1)
{
  char result; // al
  __int64 v2; // rdx
  unsigned int v3; // ecx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      {
        v3 = *(_DWORD *)(v2 + 36);
        if ( v3 <= 0xD3 )
        {
          if ( v3 > 0x9A )
            return ((1LL << ((unsigned __int8)v3 + 101)) & 0x186000000000001LL) != 0;
          if ( v3 != 11 )
            return v3 - 68 <= 3;
          return 1;
        }
        if ( v3 == 324 )
          return 1;
        if ( v3 > 0x144 )
        {
          return v3 == 376;
        }
        else
        {
          result = 1;
          if ( v3 != 282 )
            return v3 - 291 <= 1;
        }
      }
    }
  }
  return result;
}
