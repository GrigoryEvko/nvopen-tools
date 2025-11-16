// Function: sub_11D9900
// Address: 0x11d9900
//
__int64 __fastcall sub_11D9900(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 156) )
    result = *(unsigned int *)(a1 + 152);
  *(_DWORD *)(a1 + 136) = result;
  return result;
}
