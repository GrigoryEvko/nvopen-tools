// Function: sub_174C560
// Address: 0x174c560
//
__int64 __fastcall sub_174C560(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rbx
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx

  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) != 56
    || !(unsigned __int8)sub_15FA1F0(*(_QWORD *)(a2 - 24))
    || *(_BYTE *)(a2 + 16) == 72 && **(_QWORD **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)) != *(_QWORD *)v10 )
  {
    return sub_174B490(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  }
  sub_170B990(*a1, v10);
  v12 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  if ( v12 )
  {
    if ( *(_QWORD *)(a2 - 24) )
    {
      v13 = *(_QWORD *)(a2 - 16);
      v14 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v14 = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
    }
    *(_QWORD *)(a2 - 24) = v12;
    v15 = *(_QWORD *)(v12 + 8);
    *(_QWORD *)(a2 - 16) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = (a2 - 16) | *(_QWORD *)(v15 + 16) & 3LL;
    *(_QWORD *)(a2 - 8) = (v12 + 8) | *(_QWORD *)(a2 - 8) & 3LL;
    *(_QWORD *)(v12 + 8) = a2 - 24;
  }
  else if ( *(_QWORD *)(a2 - 24) )
  {
    v16 = *(_QWORD *)(a2 - 16);
    v17 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v17 = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = v17 | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(a2 - 24) = 0;
  }
  return a2;
}
