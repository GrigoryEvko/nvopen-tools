// Function: sub_2EEF4F0
// Address: 0x2eef4f0
//
_BYTE *__fastcall sub_2EEF4F0(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 v3; // r12
  _BYTE *result; // rax

  v2 = *(void **)(a1 + 32);
  v3 = a1;
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 0xEu )
  {
    v3 = sub_CB6200(a1, "- liverange:   ", 0xFu);
  }
  else
  {
    qmemcpy(v2, "- liverange:   ", 15);
    *(_QWORD *)(a1 + 32) += 15LL;
  }
  sub_2E0B3F0(a2, v3);
  result = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v3 + 24) )
    return (_BYTE *)sub_CB5D20(v3, 10);
  *(_QWORD *)(v3 + 32) = result + 1;
  *result = 10;
  return result;
}
