// Function: sub_1869D40
// Address: 0x1869d40
//
__int64 __fastcall sub_1869D40(__int64 a1)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 180) )
  {
    result = *(unsigned int *)(a1 + 176);
    *(_DWORD *)(a1 + 160) = result;
  }
  return result;
}
