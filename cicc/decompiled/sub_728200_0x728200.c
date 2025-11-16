// Function: sub_728200
// Address: 0x728200
//
__int64 __fastcall sub_728200(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 173);
  if ( !(_BYTE)result || (_BYTE)result == 12 )
  {
    *(_DWORD *)(a2 + 80) = 1;
    *(_DWORD *)(a2 + 72) = 1;
  }
  return result;
}
