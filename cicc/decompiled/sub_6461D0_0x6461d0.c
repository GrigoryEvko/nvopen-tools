// Function: sub_6461D0
// Address: 0x6461d0
//
_BOOL8 __fastcall sub_6461D0(__int64 a1, __int64 a2, int a3)
{
  _BOOL8 result; // rax
  _QWORD **v4; // rcx
  _QWORD *v5; // rdi

  result = 0;
  if ( (*(_BYTE *)(a1 + 16) & 8) != 0 && (unsigned __int8)(*(_BYTE *)(a1 + 56) - 1) <= 3u )
  {
    while ( *(_BYTE *)(a2 + 140) == 12 )
      a2 = *(_QWORD *)(a2 + 160);
    v4 = **(_QWORD ****)(a2 + 168);
    result = 0;
    if ( v4 )
    {
      v5 = *v4;
      result = 1;
      if ( *v4 )
      {
        result = 0;
        if ( a3 )
        {
          if ( !*v5 )
            return sub_646150((__int64)v5);
        }
      }
    }
  }
  return result;
}
