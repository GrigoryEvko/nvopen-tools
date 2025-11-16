// Function: sub_8D9470
// Address: 0x8d9470
//
__int64 __fastcall sub_8D9470(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_8D8C50(
             a1,
             *(__int64 (__fastcall **)(__int64, unsigned int *))(a2 + 168),
             *(_QWORD *)(a2 + 176),
             *(_DWORD *)(a2 + 184));
  if ( (_DWORD)result )
  {
    *(_DWORD *)(a2 + 80) = 1;
    *(_DWORD *)(a2 + 72) = 1;
  }
  return result;
}
