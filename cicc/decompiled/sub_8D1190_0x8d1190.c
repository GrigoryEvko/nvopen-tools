// Function: sub_8D1190
// Address: 0x8d1190
//
__int64 __fastcall sub_8D1190(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    if ( *(_QWORD *)(a1 + 8) )
    {
      *a2 = 1;
      return 1;
    }
  }
  return result;
}
