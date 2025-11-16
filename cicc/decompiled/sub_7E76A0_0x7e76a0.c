// Function: sub_7E76A0
// Address: 0x7e76a0
//
__int64 __fastcall sub_7E76A0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  _QWORD v5[10]; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned int v6; // [rsp-98h] [rbp-98h]

  result = HIDWORD(qword_4F077B4);
  if ( HIDWORD(qword_4F077B4) )
  {
    sub_76C7C0((__int64)v5);
    v5[2] = sub_7E0520;
    sub_76D560(a1, (__int64)v5, v2, v3, v4);
    return v6;
  }
  return result;
}
