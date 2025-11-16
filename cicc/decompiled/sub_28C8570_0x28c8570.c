// Function: sub_28C8570
// Address: 0x28c8570
//
__int64 __fastcall sub_28C8570(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    result = *(_QWORD *)(a1 - 32);
    if ( result )
    {
      if ( !*(_BYTE *)result
        && *(_QWORD *)(result + 24) == *(_QWORD *)(a1 + 80)
        && (*(_BYTE *)(result + 33) & 0x20) != 0
        && *(_DWORD *)(result + 36) == 336 )
      {
        return *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
