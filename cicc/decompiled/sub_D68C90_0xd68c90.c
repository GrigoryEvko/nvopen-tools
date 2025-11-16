// Function: sub_D68C90
// Address: 0xd68c90
//
__int64 __fastcall sub_D68C90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // eax
  __int64 v7; // rsi

  v4 = *(_QWORD *)(a1 + 8);
  if ( a2 )
  {
    v5 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v6 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v5 = 0;
    v6 = 0;
  }
  v7 = 0;
  if ( v6 < *(_DWORD *)(v4 + 32) )
    v7 = *(_QWORD *)(*(_QWORD *)(v4 + 24) + 8 * v5);
  return sub_103C0D0(a1, v7, a3, a4, 1, 1);
}
