// Function: sub_2D28DB0
// Address: 0x2d28db0
//
bool __fastcall sub_2D28DB0(__int64 a1, int a2, int a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx

  v5 = *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 8) + 16 * v5 - 16;
  v7 = *(_DWORD *)(v6 + 12);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 192LL) )
  {
    if ( !v7 )
    {
      v11 = sub_F03A30((__int64 *)(a1 + 8), (int)v5 - 1);
      if ( !v11 )
        return 0;
      v12 = v11;
      v13 = v11 & 0x3F;
      v14 = v12 & 0xFFFFFFFFFFFFFFC0LL;
      if ( *(_DWORD *)(v14 + 4 * v13 + 128) != a3 )
        return 0;
      return *(_DWORD *)(v14 + 8 * v13 + 4) == a2;
    }
  }
  else if ( !v7 )
  {
    return 0;
  }
  v9 = *(_QWORD *)v6;
  v10 = (unsigned int)(v7 - 1);
  if ( *(_DWORD *)(v9 + 4 * v10 + 128) != a3 )
    return 0;
  return *(_DWORD *)(v9 + 8 * v10 + 4) == a2;
}
