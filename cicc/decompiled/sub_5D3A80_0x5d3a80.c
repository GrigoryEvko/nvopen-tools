// Function: sub_5D3A80
// Address: 0x5d3a80
//
__int64 __fastcall sub_5D3A80(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 40);
  if ( (_BYTE)result == 11 )
  {
    result = *(_QWORD *)(a1 + 80);
    if ( *(_QWORD *)(result + 8) )
      *(_DWORD *)(a2 + 76) = 1;
  }
  else if ( (_BYTE)result == 16 )
  {
    result = *(_QWORD *)(a1 + 72);
    if ( result )
    {
      if ( *(_BYTE *)(result + 40) == 11 )
      {
        result = *(_QWORD *)(result + 80);
        if ( *(_QWORD *)(result + 8) )
        {
          result = sub_76CDC0(*(_QWORD *)(a1 + 48));
          *(_DWORD *)(a2 + 76) = 1;
        }
      }
    }
  }
  return result;
}
