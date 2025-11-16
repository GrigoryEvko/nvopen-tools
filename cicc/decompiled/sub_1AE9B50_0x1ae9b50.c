// Function: sub_1AE9B50
// Address: 0x1ae9b50
//
char __fastcall sub_1AE9B50(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 ***v3; // r13
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r15
  char result; // al
  __int64 v8; // rax

  v3 = *(__int64 ****)(a2 - 48);
  v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 24 * (1 - v4)) + 24LL);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 24 * (2 - v4)) + 24LL);
  if ( !(unsigned __int8)sub_1AE93B0((__int64)*v3, a1) )
    v3 = (__int64 ***)sub_1599EF0(*v3);
  result = sub_1AE8290(v5, v6, a2);
  if ( !result )
  {
    v8 = sub_15C70A0(a1 + 48);
    return sub_15A76D0(a3, (__int64)v3, v5, v6, v8, a2);
  }
  return result;
}
