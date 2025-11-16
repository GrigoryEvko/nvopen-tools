// Function: sub_8247C0
// Address: 0x8247c0
//
int __fastcall sub_8247C0(__int64 a1, __int64 a2, _DWORD *a3)
{
  int result; // eax

  result = 0;
  if ( a2 )
  {
    if ( (*(_BYTE *)(a1 + 104) & 1) != 0 )
      goto LABEL_5;
    result = dword_4D0444C;
    if ( dword_4D0444C )
      return 0;
    if ( *(_QWORD *)(a1 + 96) )
    {
LABEL_5:
      result = sub_824150(*(_QWORD **)(a1 + 88), *(_QWORD **)(a2 + 88));
      if ( result )
      {
        result = sub_824150(*(_QWORD **)(a1 + 96), *(_QWORD **)(a2 + 96));
        if ( result )
        {
          sub_6851C0(0xC00u, a3);
          return 1;
        }
      }
    }
  }
  return result;
}
