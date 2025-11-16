// Function: sub_2EEF640
// Address: 0x2eef640
//
_BYTE *__fastcall sub_2EEF640(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void *v3; // rdx
  _BYTE *result; // rax
  _QWORD v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1;
  v3 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v3 <= 0xEu )
  {
    v2 = sub_CB6200(a1, "- at:          ", 0xFu);
  }
  else
  {
    qmemcpy(v3, "- at:          ", 15);
    *(_QWORD *)(a1 + 32) += 15LL;
  }
  v5[0] = a2;
  sub_2FAD600(v5, v2);
  result = *(_BYTE **)(v2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v2 + 24) )
    return (_BYTE *)sub_CB5D20(v2, 10);
  *(_QWORD *)(v2 + 32) = result + 1;
  *result = 10;
  return result;
}
