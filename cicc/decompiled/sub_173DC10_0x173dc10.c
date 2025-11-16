// Function: sub_173DC10
// Address: 0x173dc10
//
__int64 __fastcall sub_173DC10(__int64 a1, int a2, char a3)
{
  __int64 *v4; // rax
  __int64 result; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = *(_QWORD *)(a1 + 56);
  v4 = (__int64 *)sub_16498A0(a1);
  result = sub_1563AB0(v6, v4, a2, a3);
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
