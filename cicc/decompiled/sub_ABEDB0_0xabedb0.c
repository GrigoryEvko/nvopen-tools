// Function: sub_ABEDB0
// Address: 0xabedb0
//
_BYTE *__fastcall sub_ABEDB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  _WORD *v4; // rdx
  _BYTE *result; // rax

  v2 = a1;
  v3 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v3 )
  {
    v2 = sub_CB6200(a1, "(", 1);
  }
  else
  {
    *v3 = 40;
    ++*(_QWORD *)(a1 + 32);
  }
  sub_C49420(a2, v2, 1);
  v4 = *(_WORD **)(v2 + 32);
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v4 <= 1u )
  {
    v2 = sub_CB6200(v2, ", ", 2);
  }
  else
  {
    *v4 = 8236;
    *(_QWORD *)(v2 + 32) += 2LL;
  }
  sub_C49420(a2 + 16, v2, 1);
  result = *(_BYTE **)(v2 + 32);
  if ( *(_BYTE **)(v2 + 24) == result )
    return (_BYTE *)sub_CB6200(v2, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v2 + 32);
  return result;
}
