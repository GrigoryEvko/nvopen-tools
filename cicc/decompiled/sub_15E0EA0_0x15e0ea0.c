// Function: sub_15E0EA0
// Address: 0x15e0ea0
//
__int64 __fastcall sub_15E0EA0(__int64 a1, int a2, _BYTE *a3, size_t a4)
{
  __int64 *v6; // rax
  __int64 result; // rax
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v8[0] = *(_QWORD *)(a1 + 112);
  v6 = (__int64 *)sub_15E0530(a1);
  result = sub_1563170(v8, v6, a2, a3, a4);
  *(_QWORD *)(a1 + 112) = result;
  return result;
}
