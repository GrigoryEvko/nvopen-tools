// Function: sub_15E0DF0
// Address: 0x15e0df0
//
__int64 __fastcall sub_15E0DF0(__int64 a1, int a2, char a3)
{
  __int64 *v4; // rax
  __int64 result; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = *(_QWORD *)(a1 + 112);
  v4 = (__int64 *)sub_15E0530(a1);
  result = sub_1563AB0(v6, v4, a2 + 1, a3);
  *(_QWORD *)(a1 + 112) = result;
  return result;
}
