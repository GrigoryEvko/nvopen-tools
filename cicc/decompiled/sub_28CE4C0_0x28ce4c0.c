// Function: sub_28CE4C0
// Address: 0x28ce4c0
//
unsigned __int64 __fastcall sub_28CE4C0(__int64 a1)
{
  unsigned __int64 v2; // rax
  _QWORD *v3; // rdi
  unsigned __int64 v5; // [rsp+8h] [rbp-28h] BYREF
  unsigned __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  __int64 v7[3]; // [rsp+18h] [rbp-18h] BYREF

  v2 = sub_939680(*(_QWORD **)(a1 + 56), *(_QWORD *)(a1 + 56) + 4LL * *(unsigned int *)(a1 + 52));
  v3 = *(_QWORD **)(a1 + 24);
  v5 = v2;
  v6 = sub_27B0000(v3, (__int64)&v3[*(unsigned int *)(a1 + 36)]);
  v7[0] = *(unsigned int *)(a1 + 12);
  v7[0] = sub_27B25E0(v7, (__int64 *)(a1 + 40), (__int64 *)&v6);
  return sub_C41E80(v7, (__int64 *)&v5);
}
