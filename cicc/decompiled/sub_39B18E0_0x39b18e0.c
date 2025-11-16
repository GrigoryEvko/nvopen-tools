// Function: sub_39B18E0
// Address: 0x39b18e0
//
unsigned __int64 __fastcall sub_39B18E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  if ( *(_BYTE *)(a3 + 8) == 16 )
    a3 = **(_QWORD **)(a3 + 16);
  return sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a3, a4);
}
