// Function: sub_1704330
// Address: 0x1704330
//
__int64 __fastcall sub_1704330(int a1, __int64 a2, _QWORD *a3, __int64 *a4, double a5, double a6, double a7)
{
  __int64 v7; // r12
  int v8; // eax
  __int64 *v11; // rax

  *a3 = *(_QWORD *)(a2 - 48);
  v7 = *(_QWORD *)(a2 - 24);
  *a4 = v7;
  if ( ((a1 - 11) & 0xFFFFFFFD) != 0 )
  {
    v8 = *(unsigned __int8 *)(a2 + 16);
    return (unsigned int)(v8 - 24);
  }
  v8 = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)v8 == 47 )
  {
    if ( *(_BYTE *)(v7 + 16) > 0x10u )
      return (unsigned int)(v8 - 24);
  }
  else
  {
    if ( (_BYTE)v8 != 5 )
      return (unsigned int)(v8 - 24);
    if ( *(_WORD *)(a2 + 18) != 23 )
      return (unsigned int)(v8 - 24);
    v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v7 )
      return (unsigned int)(v8 - 24);
  }
  v11 = (__int64 *)sub_15A0680(*(_QWORD *)a2, 1, 0);
  *a4 = sub_15A2D50(v11, v7, 0, 0, a5, a6, a7);
  return 15;
}
