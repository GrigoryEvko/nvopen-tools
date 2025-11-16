// Function: sub_15FDF90
// Address: 0x15fdf90
//
__int64 __fastcall sub_15FDF90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v6; // rdx
  int v7; // eax

  v4 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    v4 = **(_QWORD **)(v4 + 16);
  v6 = a2;
  v7 = *(_DWORD *)(v4 + 8) >> 8;
  if ( *(_BYTE *)(a2 + 8) == 16 )
    v6 = **(_QWORD **)(a2 + 16);
  if ( *(_DWORD *)(v6 + 8) >> 8 == v7 )
    return sub_15FDBD0(47, a1, a2, a3, a4);
  else
    return sub_15FDBD0(48, a1, a2, a3, a4);
}
