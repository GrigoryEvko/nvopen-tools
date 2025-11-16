// Function: sub_1E85AA0
// Address: 0x1e85aa0
//
_BYTE *__fastcall sub_1E85AA0(__int64 a1)
{
  void *v1; // rax
  __int64 v2; // r12
  _BYTE *result; // rax
  _QWORD v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_16E8CB0();
  v4[0] = a1;
  v2 = sub_1263B40((__int64)v1, "- at:          ");
  sub_1F10810(v4, v2);
  result = *(_BYTE **)(v2 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v2 + 16) )
    return (_BYTE *)sub_16E7DE0(v2, 10);
  *(_QWORD *)(v2 + 24) = result + 1;
  *result = 10;
  return result;
}
