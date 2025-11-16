// Function: sub_AE9D00
// Address: 0xae9d00
//
__int64 __fastcall sub_AE9D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v6; // [rsp+0h] [rbp-30h]

  v6 = sub_9208B0(a2, *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL));
  sub_AE6F30(a1, a2, *(_QWORD *)(a3 - 32), v6, v4);
  return a1;
}
