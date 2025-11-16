// Function: sub_B52190
// Address: 0xb52190
//
__int64 __fastcall sub_B52190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  __int64 v5; // r10
  __int64 v6; // rax
  int v7; // eax

  v5 = a2;
  v6 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v7 = *(_DWORD *)(v6 + 8) >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
    a2 = **(_QWORD **)(a2 + 16);
  if ( *(_DWORD *)(a2 + 8) >> 8 == v7 )
    return sub_B51D30(49, a1, v5, a3, a4, a5);
  else
    return sub_B51D30(50, a1, v5, a3, a4, a5);
}
