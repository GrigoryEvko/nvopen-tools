// Function: sub_728F20
// Address: 0x728f20
//
__int64 __fastcall sub_728F20(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( result && (*(_BYTE *)(result + 194) & 8) == 0
    || *(_BYTE *)(a1 + 48) == 5 && (result = *(_QWORD *)(a1 + 56)) != 0 && (*(_BYTE *)(result + 194) & 6) == 0 )
  {
    *(_DWORD *)(a2 + 80) = 1;
    *(_DWORD *)(a2 + 72) = 1;
  }
  return result;
}
