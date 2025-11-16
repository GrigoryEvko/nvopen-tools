// Function: sub_39B5410
// Address: 0x39b5410
//
__int64 __fastcall sub_39B5410(__int64 a1, unsigned int a2, _QWORD *a3, __int64 a4)
{
  __int64 v5; // rdx
  int v7; // r12d

  v5 = a4;
  if ( *(_BYTE *)(a4 + 8) == 16 )
    v5 = **(_QWORD **)(a4 + 16);
  v7 = sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v5, a4);
  return v7 + (unsigned int)sub_39B4D00((__int64 *)(a1 + 8), a2, a3, *(_QWORD **)(a4 + 24), 0);
}
