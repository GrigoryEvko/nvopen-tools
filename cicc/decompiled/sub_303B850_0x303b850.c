// Function: sub_303B850
// Address: 0x303b850
//
__int64 __fastcall sub_303B850(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  __int16 v6; // ax

  v6 = *(_WORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  if ( v6 == 12 )
    return sub_303AE60(a1, a2, a3, a4, a5, a6);
  if ( v6 != 13 )
    BUG();
  return sub_303B360(a1, a2, a3, a4);
}
