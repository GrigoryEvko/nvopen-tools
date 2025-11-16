// Function: sub_BC5D40
// Address: 0xbc5d40
//
char __fastcall sub_BC5D40(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  char result; // al
  __int64 v4; // rbx
  _QWORD v5[5]; // [rsp-28h] [rbp-28h] BYREF

  result = qword_4F83388;
  if ( !(_BYTE)qword_4F83388 )
  {
    v5[4] = v2;
    v4 = qword_4F83570;
    v5[0] = a1;
    v5[1] = a2;
    return v4 != sub_BC5B40(qword_4F83568, qword_4F83570, (__int64)v5);
  }
  return result;
}
