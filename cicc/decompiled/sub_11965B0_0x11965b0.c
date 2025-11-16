// Function: sub_11965B0
// Address: 0x11965b0
//
__int64 __fastcall sub_11965B0(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdx
  _BYTE *v10; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) || (unsigned __int8)(*(_BYTE *)v4 - 55) > 1u )
    return 0;
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(v4 - 8);
  else
    v7 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)(v7 + 32) != *(_QWORD *)(a1 + 8) )
    return 0;
  v8 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v8 == 17 )
  {
    **(_QWORD **)(a1 + 16) = v8 + 24;
    return 1;
  }
  else
  {
    v9 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v9 > 1 )
      return 0;
    if ( *(_BYTE *)v8 > 0x15u )
      return 0;
    v10 = sub_AD7630(v8, *(unsigned __int8 *)(a1 + 24), v9);
    if ( !v10 || *v10 != 17 )
      return 0;
    **(_QWORD **)(a1 + 16) = v10 + 24;
    return 1;
  }
}
