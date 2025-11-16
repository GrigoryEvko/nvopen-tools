// Function: sub_7F52E0
// Address: 0x7f52e0
//
__int64 sub_7F52E0()
{
  __int64 result; // rax
  _QWORD *v1; // r13
  __int64 v2; // r12
  __int64 v3; // rax
  _QWORD *v4; // rax

  result = qword_4F18B18;
  if ( !qword_4F18B18 )
  {
    v1 = sub_72BA30(byte_4F06A51[0]);
    v2 = sub_7E1C10();
    v3 = sub_72CBE0();
    v4 = sub_7F5250(v3, v2, (__int64)v1);
    result = sub_72D2E0(v4);
    qword_4F18B18 = result;
  }
  return result;
}
