// Function: sub_728910
// Address: 0x728910
//
__int64 __fastcall sub_728910(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a1 && *(_BYTE *)(a1 + 24) == 3 )
  {
    result = *(_QWORD *)(a1 + 56);
    if ( (*(_BYTE *)(result - 8) & 1) == 0 )
    {
      qword_4F07A40 = a1;
      *(_DWORD *)(a2 + 80) = 1;
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
  return result;
}
