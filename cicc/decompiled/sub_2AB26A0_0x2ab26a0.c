// Function: sub_2AB26A0
// Address: 0x2ab26a0
//
__int64 __fastcall sub_2AB26A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  char v7; // r8

  result = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)result == -1 )
  {
    v7 = sub_F6E590(*(_QWORD *)(a1 + 104), a2, a3, a4, a5, a6);
    result = 0;
    if ( !v7 )
      return *(unsigned int *)(a1 + 40);
  }
  return result;
}
