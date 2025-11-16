// Function: sub_2EEF5A0
// Address: 0x2eef5a0
//
_BYTE *__fastcall sub_2EEF5A0(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // rdi
  _BYTE *result; // rax

  v2 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 0xEu )
  {
    a1 = sub_CB6200(a1, "- segment:     ", 0xFu);
  }
  else
  {
    qmemcpy(v2, "- segment:     ", 15);
    *(_QWORD *)(a1 + 32) += 15LL;
  }
  v3 = sub_2E0B2D0(a1, a2);
  result = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v3 + 24) )
    return (_BYTE *)sub_CB5D20(v3, 10);
  *(_QWORD *)(v3 + 32) = result + 1;
  *result = 10;
  return result;
}
