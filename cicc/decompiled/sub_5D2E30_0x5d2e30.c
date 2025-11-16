// Function: sub_5D2E30
// Address: 0x5d2e30
//
__int64 __fastcall sub_5D2E30(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 140) == 12 && *(char *)(a1 + 186) < 0 )
  {
    *a2 = 1;
    return 1;
  }
  return result;
}
