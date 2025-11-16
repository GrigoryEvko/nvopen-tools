// Function: sub_15D3940
// Address: 0x15d3940
//
__int64 __fastcall sub_15D3940(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = a1 + 160;
  *(_QWORD *)(v2 + 64) = a2;
  sub_15D3930(v2);
  return 0;
}
