// Function: sub_1F15AD0
// Address: 0x1f15ad0
//
__int64 __fastcall sub_1F15AD0(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rcx
  unsigned int v7; // r8d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int16 v11; // ax

  v6 = a3;
  v7 = 1;
  v8 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v9 = *(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 != v9 )
    return v7;
  if ( !(_BYTE)v6 )
    return 0;
  if ( *(_BYTE *)(a2 + 32) )
  {
    v7 = *(unsigned __int8 *)(a2 + 33);
    if ( (_BYTE)v7 )
      return v7;
  }
  if ( !v8 )
    BUG();
  v11 = **(_WORD **)(*(_QWORD *)(v8 + 16) + 16LL);
  if ( v11 == 15 || v11 == 10 )
    return 0;
  else
    return sub_1F15030(a1, *(_QWORD *)(a2 + 8), v9, v6, v7, a6);
}
