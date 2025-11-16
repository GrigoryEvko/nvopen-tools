// Function: sub_2D44290
// Address: 0x2d44290
//
__int64 __fastcall sub_2D44290(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD v5[3]; // [rsp+0h] [rbp-20h] BYREF

  v1 = sub_B43CC0(a1);
  v2 = sub_9208B0(v1, *(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL));
  v5[1] = v3;
  v5[0] = (unsigned __int64)(v2 + 7) >> 3;
  return sub_CA1930(v5);
}
