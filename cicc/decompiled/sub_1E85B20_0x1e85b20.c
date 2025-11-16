// Function: sub_1E85B20
// Address: 0x1e85b20
//
_BYTE *__fastcall sub_1E85B20(__int64 a1)
{
  void *v1; // rax
  __int64 v2; // r12
  _BYTE *result; // rax

  v1 = sub_16E8CB0();
  v2 = sub_1263B40((__int64)v1, "- interval:    ");
  sub_1DB53F0(a1, v2);
  result = *(_BYTE **)(v2 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v2 + 16) )
    return (_BYTE *)sub_16E7DE0(v2, 10);
  *(_QWORD *)(v2 + 24) = result + 1;
  *result = 10;
  return result;
}
