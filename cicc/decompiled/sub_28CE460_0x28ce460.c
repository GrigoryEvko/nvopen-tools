// Function: sub_28CE460
// Address: 0x28ce460
//
unsigned __int64 __fastcall sub_28CE460(__int64 a1)
{
  unsigned __int64 v2; // [rsp+0h] [rbp-20h] BYREF
  __int64 v3[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_27B0000(*(_QWORD **)(a1 + 24), *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 36));
  v3[0] = *(unsigned int *)(a1 + 12);
  v3[0] = sub_27B25E0(v3, (__int64 *)(a1 + 40), (__int64 *)&v2);
  return sub_28CE130(v3, (__int64 *)(a1 + 48));
}
