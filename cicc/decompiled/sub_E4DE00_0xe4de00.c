// Function: sub_E4DE00
// Address: 0xe4de00
//
_BYTE *__fastcall sub_E4DE00(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( (unsigned __int8)sub_E81180(a2, v6) )
    return (_BYTE *)sub_E98EB0(a1, v6[0], 0);
  v3 = *(_QWORD *)(a1 + 304);
  v4 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 9u )
  {
    sub_CB6200(v3, "\t.uleb128 ", 0xAu);
  }
  else
  {
    qmemcpy(v4, "\t.uleb128 ", 10);
    *(_QWORD *)(v3 + 32) += 10LL;
  }
  sub_E7FAD0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
  return sub_E4D880(a1);
}
