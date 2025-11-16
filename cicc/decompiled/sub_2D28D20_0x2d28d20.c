// Function: sub_2D28D20
// Address: 0x2d28d20
//
bool __fastcall sub_2D28D20(__int64 a1, int a2, int a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v11; // rax
  _DWORD *v12; // rax

  v5 = *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 8) + 16 * v5 - 16;
  v7 = *(_DWORD *)(v6 + 8);
  v8 = (unsigned int)(*(_DWORD *)(v6 + 12) + 1);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 192LL) )
  {
    if ( (unsigned int)v8 >= v7 )
    {
      v11 = sub_F03C90((__int64 *)(a1 + 8), (int)v5 - 1);
      if ( !v11 )
        return 0;
      v12 = (_DWORD *)(v11 & 0xFFFFFFFFFFFFFFC0LL);
      if ( v12[32] != a3 )
        return 0;
      return *v12 == a2;
    }
  }
  else if ( (unsigned int)v8 >= v7 )
  {
    return 0;
  }
  v9 = *(_QWORD *)v6;
  if ( *(_DWORD *)(v9 + 4 * v8 + 128) != a3 )
    return 0;
  return *(_DWORD *)(v9 + 8 * v8) == a2;
}
