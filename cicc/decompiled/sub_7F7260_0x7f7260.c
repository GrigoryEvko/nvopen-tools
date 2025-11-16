// Function: sub_7F7260
// Address: 0x7f7260
//
__int64 __fastcall sub_7F7260(__int64 a1, __int64 a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v6; // rdi

  *a3 = 0;
  *a4 = 0;
  v6 = *(_QWORD *)(a1 + 32);
  if ( v6 == a2 )
    qword_4D03F68[5] = v6;
  else
    sub_7F7260();
  if ( (*(_BYTE *)(a1 + 49) & 8) == 0 || !*(_QWORD *)(a1 + 32) )
    *a4 = 1;
  if ( (*(_BYTE *)(a1 + 50) & 1) == 0 )
    qword_4D03F68[5] = a1;
  return sub_7E18B0();
}
