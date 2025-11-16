// Function: sub_1E85B90
// Address: 0x1e85b90
//
_BYTE *__fastcall sub_1E85B90(__int64 a1)
{
  void *v1; // rax
  __int64 v2; // rax
  __int64 v3; // rdi
  _BYTE *result; // rax

  v1 = sub_16E8CB0();
  v2 = sub_1263B40((__int64)v1, "- segment:     ");
  v3 = sub_1DB4FB0(v2, a1);
  result = *(_BYTE **)(v3 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v3 + 16) )
    return (_BYTE *)sub_16E7DE0(v3, 10);
  *(_QWORD *)(v3 + 24) = result + 1;
  *result = 10;
  return result;
}
