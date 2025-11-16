// Function: sub_CB1E40
// Address: 0xcb1e40
//
_BYTE *__fastcall sub_CB1E40(__int64 a1)
{
  __int64 v2; // rdi
  _BYTE *result; // rax

  v2 = *(_QWORD *)(a1 + 16);
  result = *(_BYTE **)(v2 + 32);
  if ( *(_BYTE **)(v2 + 24) == result )
  {
    result = (_BYTE *)sub_CB6200(v2, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v2 + 32);
  }
  *(_DWORD *)(a1 + 80) = 0;
  return result;
}
