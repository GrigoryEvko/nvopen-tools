// Function: sub_28CE3C0
// Address: 0x28ce3c0
//
unsigned __int64 __fastcall sub_28CE3C0(__int64 a1)
{
  __int64 v2; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v3; // [rsp+8h] [rbp-18h] BYREF

  v3 = sub_27B0000(*(_QWORD **)(a1 + 24), *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 36));
  v2 = *(unsigned int *)(a1 + 12);
  return sub_27B25E0(&v2, (__int64 *)(a1 + 40), (__int64 *)&v3);
}
