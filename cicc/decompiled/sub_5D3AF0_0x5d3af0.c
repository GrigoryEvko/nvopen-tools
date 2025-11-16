// Function: sub_5D3AF0
// Address: 0x5d3af0
//
__int64 __fastcall sub_5D3AF0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 140) == 7 )
  {
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 168) + 48LL) )
    {
      *a2 = 1;
      return 1;
    }
  }
  return result;
}
