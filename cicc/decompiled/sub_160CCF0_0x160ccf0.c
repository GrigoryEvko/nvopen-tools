// Function: sub_160CCF0
// Address: 0x160ccf0
//
__int64 __fastcall sub_160CCF0(__int64 a1)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 180) )
  {
    result = *(unsigned int *)(a1 + 176);
    *(_DWORD *)(a1 + 160) = result;
  }
  return result;
}
