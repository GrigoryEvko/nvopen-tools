// Function: sub_2BDC3A0
// Address: 0x2bdc3a0
//
_BYTE *__fastcall sub_2BDC3A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  _BYTE *result; // rax

  v2 = sub_CB6200(a2, *(unsigned __int8 **)(a1 + 8), *(_QWORD *)(a1 + 16));
  result = *(_BYTE **)(v2 + 32);
  if ( *(_BYTE **)(v2 + 24) == result )
    return (_BYTE *)sub_CB6200(v2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v2 + 32);
  return result;
}
