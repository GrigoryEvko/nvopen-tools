// Function: sub_1341E90
// Address: 0x1341e90
//
_QWORD *__fastcall sub_1341E90(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  _QWORD *result; // rax
  __int64 v6; // [rsp+0h] [rbp-1B0h]
  _QWORD *v7; // [rsp+10h] [rbp-1A0h] BYREF
  _QWORD *v8; // [rsp+18h] [rbp-198h] BYREF
  _QWORD v9[50]; // [rsp+20h] [rbp-190h] BYREF

  v4 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v6 = a3;
    sub_130D500(v9);
    a3 = v6;
    v4 = v9;
  }
  sub_1341260(a1, a2, v4, a3, 1, 0, (__int64 *)&v7, (unsigned __int64 *)&v8);
  result = v8;
  *v7 = 0xE8000000000000LL;
  if ( result )
    *result = 0xE8000000000000LL;
  return result;
}
