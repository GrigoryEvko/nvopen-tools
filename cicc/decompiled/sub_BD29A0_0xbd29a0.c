// Function: sub_BD29A0
// Address: 0xbd29a0
//
__int64 __fastcall sub_BD29A0(__int64 a1)
{
  __int64 result; // rax
  unsigned int v2; // eax
  bool v3; // cf
  bool v4; // zf

  result = 0;
  if ( a1 )
  {
    if ( *(_BYTE *)a1 == 85 )
    {
      result = *(_QWORD *)(a1 - 32);
      if ( result )
      {
        if ( !*(_BYTE *)result
          && *(_QWORD *)(result + 24) == *(_QWORD *)(a1 + 80)
          && (*(_BYTE *)(result + 33) & 0x20) != 0 )
        {
          v2 = *(_DWORD *)(result + 36);
          if ( v2 > 0x45 )
          {
            v4 = v2 == 71;
            result = 0;
            if ( v4 )
              return a1;
          }
          else
          {
            v3 = v2 < 0x44;
            result = 0;
            if ( !v3 )
              return a1;
          }
        }
        else
        {
          return 0;
        }
      }
    }
  }
  return result;
}
