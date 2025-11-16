// Function: sub_37F95F0
// Address: 0x37f95f0
//
_QWORD *__fastcall sub_37F95F0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  bool v7; // cc

  v6 = 0;
  v7 = a2[5] <= 0xBu;
  a2[4] = 0;
  if ( v7 )
  {
    sub_C8D290((__int64)(a2 + 3), a2 + 6, 12, 1u, a5, a6);
    v6 = a2[4];
  }
  qmemcpy((void *)(a2[3] + v6), "<field list>", 12);
  a2[4] += 12LL;
  *a1 = 1;
  return a1;
}
