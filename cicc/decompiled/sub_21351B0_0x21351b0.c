// Function: sub_21351B0
// Address: 0x21351b0
//
__int64 *__fastcall sub_21351B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  __int64 v6; // [rsp+8h] [rbp-28h]
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  int v8; // [rsp+18h] [rbp-18h]

  v2 = *(_QWORD *)(a2 + 32);
  LODWORD(v6) = 0;
  v8 = 0;
  v5 = 0;
  v3 = *(_QWORD *)(v2 + 48);
  v7 = 0;
  sub_20174B0(a1, *(_QWORD *)(v2 + 40), v3, &v5, &v7);
  return sub_1D2DF70(
           *(_QWORD **)(a1 + 8),
           (__int64 *)a2,
           **(_QWORD **)(a2 + 32),
           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
           v5,
           v6);
}
