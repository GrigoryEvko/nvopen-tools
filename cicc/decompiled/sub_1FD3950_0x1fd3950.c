// Function: sub_1FD3950
// Address: 0x1fd3950
//
__int64 __fastcall sub_1FD3950(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax

  v4 = *(_QWORD *)(a1 + 56);
  v5 = (__int64)sub_1E0B640(v4, a4, a3, 0);
  sub_1DD5BA0((__int64 *)(a1 + 16), v5);
  v6 = *a2;
  v7 = *(_QWORD *)v5;
  *(_QWORD *)(v5 + 8) = a2;
  v6 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v6 | v7 & 7;
  *(_QWORD *)(v6 + 8) = v5;
  *a2 = v5 | *a2 & 7;
  return v4;
}
