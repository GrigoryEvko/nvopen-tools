// Function: sub_36FFC90
// Address: 0x36ffc90
//
_BYTE *__fastcall sub_36FFC90(__int64 *a1)
{
  _BYTE *result; // rax
  __int64 v2; // rbx
  __int64 i; // r13
  __int64 v5; // rdi

  result = (_BYTE *)*a1;
  v2 = *(_QWORD *)(*a1 + 8);
  for ( i = *(_QWORD *)(*a1 + 16); i != v2; result = sub_310D630(v5, a1[1]) )
  {
    v5 = v2;
    v2 += 80;
  }
  return result;
}
