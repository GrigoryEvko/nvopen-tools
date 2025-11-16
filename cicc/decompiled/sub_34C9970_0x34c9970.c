// Function: sub_34C9970
// Address: 0x34c9970
//
__int64 __fastcall sub_34C9970(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_WORD *)(a1 + 68) == 3 )
    return *(_DWORD *)(a1 + 44) & 1;
  return result;
}
