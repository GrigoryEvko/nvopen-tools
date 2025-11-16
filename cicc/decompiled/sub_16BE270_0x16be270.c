// Function: sub_16BE270
// Address: 0x16be270
//
__int64 __fastcall sub_16BE270(__int64 a1, int a2)
{
  int v2; // ebx
  __int64 v3; // rsi

  sub_16BE1E0(a1, *(char **)(a1 + 8), *(_QWORD *)(a1 + 24) - *(_QWORD *)(a1 + 8));
  v2 = a2 - *(_DWORD *)(a1 + 48);
  v3 = (unsigned int)v2;
  if ( v2 <= 0 )
    v3 = 1;
  sub_16E8750(a1, v3);
  return a1;
}
