// Function: sub_1342700
// Address: 0x1342700
//
unsigned __int64 __fastcall sub_1342700(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  _QWORD *v6; // rcx
  __int64 *v7; // rsi

  v6 = *(_QWORD **)(a3 + 8);
  if ( v6 )
    *v6 = 0xE8000000000000LL;
  v7 = *(__int64 **)(a3 + 16);
  if ( *(_QWORD *)(a3 + 24) )
  {
    *v7 = 0xE8000000000000LL;
    v7 = *(__int64 **)(a3 + 24);
  }
  return sub_1341500(*(__int64 **)a3, v7, a4, 232, 0);
}
