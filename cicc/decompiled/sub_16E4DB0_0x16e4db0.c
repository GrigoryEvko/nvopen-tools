// Function: sub_16E4DB0
// Address: 0x16e4db0
//
_BYTE *__fastcall sub_16E4DB0(__int64 a1)
{
  __int64 v2; // rdi
  _BYTE *result; // rax

  v2 = *(_QWORD *)(a1 + 16);
  result = *(_BYTE **)(v2 + 24);
  if ( *(_BYTE **)(v2 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v2, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v2 + 24);
  }
  *(_DWORD *)(a1 + 80) = 0;
  return result;
}
