// Function: sub_15F4220
// Address: 0x15f4220
//
char __fastcall sub_15F4220(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // r12
  char v12; // r11
  __int64 v13; // rax
  char v14; // r10
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r8
  __int64 v19; // r8
  __int64 v20; // rdx

  if ( *(_BYTE *)(a1 + 16) != *(_BYTE *)(a2 + 16) )
    return 0;
  v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v4 != (_DWORD)v5 )
    return 0;
  v7 = *(_QWORD *)a2;
  v8 = *(_QWORD *)a1;
  if ( (a3 & 2) != 0 )
  {
    if ( *(_BYTE *)(v8 + 8) == 16 )
      v8 = **(_QWORD **)(v8 + 16);
    if ( *(_BYTE *)(v7 + 8) == 16 )
      v7 = **(_QWORD **)(v7 + 16);
  }
  if ( v8 != v7 )
    return 0;
  if ( (_DWORD)v4 )
  {
    v9 = 24 * v4;
    v10 = a2 - 24 * v5;
    v11 = a1 - 24 * v4;
    v12 = *(_BYTE *)(a1 + 23) & 0x40;
    v13 = 0;
    v14 = *(_BYTE *)(a2 + 23) & 0x40;
    do
    {
      v15 = v11;
      if ( (a3 & 2) != 0 )
      {
        if ( v12 )
          v15 = *(_QWORD *)(a1 - 8);
        v16 = **(_QWORD **)(v15 + v13);
        if ( *(_BYTE *)(v16 + 8) == 16 )
          v16 = **(_QWORD **)(v16 + 16);
        v17 = v10;
        if ( v14 )
          v17 = *(_QWORD *)(a2 - 8);
        v18 = **(_QWORD **)(v17 + v13);
        if ( *(_BYTE *)(v18 + 8) == 16 )
          v18 = **(_QWORD **)(v18 + 16);
        if ( v18 != v16 )
          return 0;
      }
      else
      {
        if ( v12 )
          v15 = *(_QWORD *)(a1 - 8);
        v19 = **(_QWORD **)(v15 + v13);
        v20 = v10;
        if ( v14 )
          v20 = *(_QWORD *)(a2 - 8);
        if ( **(_QWORD **)(v20 + v13) != v19 )
          return 0;
      }
      v13 += 24;
    }
    while ( v9 != v13 );
  }
  return sub_15F3E20(a1, a2, a3 & 1);
}
