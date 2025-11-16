// Function: sub_80A070
// Address: 0x80a070
//
_BOOL8 __fastcall sub_80A070(__int64 a1, _DWORD *a2)
{
  _BOOL8 result; // rax
  char v3; // cl

  *a2 = 0;
  if ( !*(_QWORD *)(a1 + 8) )
    return *(_BYTE *)(a1 + 174) == 1;
  result = 0;
  if ( unk_4F07290 != a1 && (*(_BYTE *)(a1 - 8) & 0x10) == 0 )
  {
    v3 = *(_BYTE *)(a1 + 174);
    if ( v3 != 7 )
    {
      result = 1;
      if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x30 )
      {
        if ( (*(_QWORD *)(a1 + 192) & 0x20000000002000LL) == 0x20000000002000LL )
        {
          *(_BYTE *)(a1 + 197) = *(_BYTE *)(a1 + 197) & 0x9F | 0x20;
        }
        else
        {
          result = 0;
          if ( v3 )
          {
            *a2 = 1;
            return 1;
          }
        }
      }
    }
  }
  return result;
}
