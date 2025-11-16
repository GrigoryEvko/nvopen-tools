// Function: sub_17315C0
// Address: 0x17315c0
//
__int64 __fastcall sub_17315C0(_QWORD **a1, __int64 a2)
{
  int v2; // eax
  int v4; // eax
  __int64 *v5; // rsi
  __int64 v6; // rax
  char v7; // dl
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    v4 = v2 - 24;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *(unsigned __int16 *)(a2 + 18);
  }
  if ( v4 != 36 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v6 = *v5;
  v7 = *(_BYTE *)(*v5 + 16);
  if ( v7 == 50 )
  {
    v10 = *(_QWORD *)(v6 - 48);
    if ( !v10 )
      return 0;
    **a1 = v10;
    v9 = *(_QWORD *)(v6 - 24);
    if ( *(_BYTE *)(v9 + 16) != 13 )
      return 0;
  }
  else
  {
    if ( v7 != 5 )
      return 0;
    if ( *(_WORD *)(v6 + 18) != 26 )
      return 0;
    v8 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    if ( !v8 )
      return 0;
    **a1 = v8;
    v9 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v9 + 16) != 13 )
      return 0;
  }
  *a1[1] = v9;
  return 1;
}
