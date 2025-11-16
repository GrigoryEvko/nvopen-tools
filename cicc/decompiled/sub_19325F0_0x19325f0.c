// Function: sub_19325F0
// Address: 0x19325f0
//
unsigned __int64 __fastcall sub_19325F0(__int64 a1)
{
  __int64 v2; // [rsp+0h] [rbp-20h] BYREF
  __int64 v3[3]; // [rsp+8h] [rbp-18h] BYREF

  v3[0] = sub_1930F10(*(_QWORD **)(a1 + 24), *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 36));
  v2 = *(unsigned int *)(a1 + 12);
  v3[0] = sub_1930D90(&v2, (__int64 *)(a1 + 40), v3);
  return sub_1930E50(v3, (int *)(a1 + 48), (char *)(a1 + 52));
}
