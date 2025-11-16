// Function: sub_28AD070
// Address: 0x28ad070
//
__int64 __fastcall sub_28AD070(__int64 a1)
{
  __int64 result; // rax
  bool v2; // zf

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
          v2 = ((*(_DWORD *)(result + 36) - 238) & 0xFFFFFFFD) == 0;
          result = 0;
          if ( v2 )
            return a1;
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
