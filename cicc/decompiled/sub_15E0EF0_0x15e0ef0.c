// Function: sub_15E0EF0
// Address: 0x15e0ef0
//
__int64 __fastcall sub_15E0EF0(__int64 a1, int a2, _QWORD *a3)
{
  __int64 *v4; // rax
  __int64 result; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = *(_QWORD *)(a1 + 112);
  v4 = (__int64 *)sub_15E0530(a1);
  result = sub_1563330(v6, v4, a2, a3);
  *(_QWORD *)(a1 + 112) = result;
  return result;
}
