// Function: sub_15E0DA0
// Address: 0x15e0da0
//
__int64 __fastcall sub_15E0DA0(__int64 a1, __int32 a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 result; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = *(_QWORD *)(a1 + 112);
  v4 = (__int64 *)sub_15E0530(a1);
  result = sub_1563D60(v6, v4, a2, a3);
  *(_QWORD *)(a1 + 112) = result;
  return result;
}
