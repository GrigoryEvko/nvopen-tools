// Function: sub_5E8C40
// Address: 0x5e8c40
//
__int64 __fastcall sub_5E8C40(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 24) == 24 )
  {
    result = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)result )
    {
      *(_DWORD *)(a2 + 80) = 1;
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
  return result;
}
