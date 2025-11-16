// Function: sub_B46690
// Address: 0xb46690
//
__int64 __fastcall sub_B46690(__int64 a1)
{
  __int64 result; // rax
  unsigned int v2; // eax

  if ( *(_BYTE *)a1 == 34 || (unsigned __int8)(*(_BYTE *)a1 - 61) > 0x18u )
    return 0;
  switch ( *(_BYTE *)a1 )
  {
    case '=':
    case 'B':
      return *(_QWORD *)(a1 + 8);
    case '>':
      return *(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL);
    case 'A':
      return *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
    case 'U':
      result = 0;
      if ( *(_BYTE *)a1 != 85 )
        return result;
      result = *(_QWORD *)(a1 - 32);
      if ( !result )
        return result;
      if ( *(_BYTE *)result || *(_QWORD *)(result + 24) != *(_QWORD *)(a1 + 80) || (*(_BYTE *)(result + 33) & 0x20) == 0 )
        return 0;
      v2 = *(_DWORD *)(result + 36);
      if ( v2 > 0xE6 )
      {
        if ( v2 != 438 )
        {
          if ( v2 > 0x1B6 )
          {
            if ( v2 == 470 || v2 == 481 )
              return *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
            return 0;
          }
          if ( v2 != 433 )
            return 0;
        }
        return *(_QWORD *)(a1 + 8);
      }
      if ( v2 > 0xE4 || v2 == 225 )
        return *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
      if ( v2 > 0xE1 || v2 == 167 )
        return *(_QWORD *)(a1 + 8);
      if ( v2 == 168 )
        return *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
      return 0;
    default:
      return 0;
  }
}
