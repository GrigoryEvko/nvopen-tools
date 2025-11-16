// Function: sub_85F0B0
// Address: 0x85f0b0
//
__int64 *__fastcall sub_85F0B0(unsigned __int8 a1, __int64 a2, int a3)
{
  int v4; // esi
  __int64 *v6; // r13
  __int64 v7; // r12
  __int64 i; // rbx

  v4 = -1;
  if ( (unsigned __int8)(a1 - 4) <= 1u )
    v4 = *(_DWORD *)(*(_QWORD *)(a2 + 128) + 24LL);
  v6 = sub_85C120(a1, v4, 0, 0, a2, 0, 0, 0, 0, 0, 0, 0, 0);
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( a3 != -1 )
    *(_DWORD *)(v7 + 552) = a3;
  for ( i = *(_QWORD *)(*(_QWORD *)(a2 + 128) + 184LL); i; i = *(_QWORD *)i )
    sub_85EE10(i, v7, *(_DWORD *)(i + 56));
  return v6;
}
