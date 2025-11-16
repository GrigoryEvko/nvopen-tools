// Function: sub_3700640
// Address: 0x3700640
//
_BYTE *__fastcall sub_3700640(__int64 *a1)
{
  __int64 v1; // rdi
  _BYTE *result; // rax

  v1 = *a1;
  result = *(_BYTE **)(v1 + 32);
  if ( *(_BYTE **)(v1 + 24) == result )
    return (_BYTE *)sub_CB6200(v1, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v1 + 32);
  return result;
}
