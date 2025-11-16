// Function: sub_BD9900
// Address: 0xbd9900
//
_BYTE *__fastcall sub_BD9900(__int64 *a1, const char *a2)
{
  __int64 v3; // rdi
  _BYTE *result; // rax

  sub_A62C00(a2, *a1, (__int64)(a1 + 2), a1[1]);
  v3 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 24) )
    return (_BYTE *)sub_CB5D20(v3, 10);
  *(_QWORD *)(v3 + 32) = result + 1;
  *result = 10;
  return result;
}
