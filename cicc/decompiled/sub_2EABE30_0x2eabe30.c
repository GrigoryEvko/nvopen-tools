// Function: sub_2EABE30
// Address: 0x2eabe30
//
_BYTE *__fastcall sub_2EABE30(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // r12
  _BYTE *result; // rax

  v2 = *(void **)(a1 + 32);
  v3 = a1;
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 9u )
  {
    v3 = sub_CB6200(a1, "<mcsymbol ", 0xAu);
  }
  else
  {
    qmemcpy(v2, "<mcsymbol ", 10);
    *(_QWORD *)(a1 + 32) += 10LL;
  }
  sub_EA12C0(a2, v3, 0);
  result = *(_BYTE **)(v3 + 32);
  if ( *(_BYTE **)(v3 + 24) == result )
    return (_BYTE *)sub_CB6200(v3, (unsigned __int8 *)">", 1u);
  *result = 62;
  ++*(_QWORD *)(v3 + 32);
  return result;
}
