// Function: sub_8AD0D0
// Address: 0x8ad0d0
//
_DWORD *__fastcall sub_8AD0D0(__int64 a1, int a2, char a3)
{
  _DWORD *result; // rax
  __int64 v4; // rdi

  result = &dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    v4 = *(_QWORD *)(a1 + 96);
    if ( v4 )
      return (_DWORD *)sub_8AC530(v4, a2, a3);
  }
  return result;
}
