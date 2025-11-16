// Function: sub_728430
// Address: 0x728430
//
__int64 __fastcall sub_728430(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_8D97B0(a1);
  if ( (_DWORD)result )
  {
    *(_DWORD *)(a2 + 80) = 1;
    *(_DWORD *)(a2 + 72) = 1;
  }
  return result;
}
