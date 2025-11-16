// Function: sub_1E31EA0
// Address: 0x1e31ea0
//
_BYTE *__fastcall sub_1E31EA0(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // r12
  _BYTE *result; // rax

  v2 = *(void **)(a1 + 24);
  v3 = a1;
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v2 <= 9u )
  {
    v3 = sub_16E7EE0(a1, "<mcsymbol ", 0xAu);
  }
  else
  {
    qmemcpy(v2, "<mcsymbol ", 10);
    *(_QWORD *)(a1 + 24) += 10LL;
  }
  sub_38E2490(a2, v3, 0);
  result = *(_BYTE **)(v3 + 24);
  if ( *(_BYTE **)(v3 + 16) == result )
    return (_BYTE *)sub_16E7EE0(v3, ">", 1u);
  *result = 62;
  ++*(_QWORD *)(v3 + 24);
  return result;
}
