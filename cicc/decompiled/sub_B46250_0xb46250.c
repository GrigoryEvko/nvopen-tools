// Function: sub_B46250
// Address: 0xb46250
//
__int64 __fastcall sub_B46250(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r14
  char v10; // r12
  __int64 v11; // rax
  char v12; // r11
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r9
  __int64 v16; // r9
  __int64 v17; // r8

  if ( *(_BYTE *)a1 != *(_BYTE *)a2 )
    return 0;
  v3 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (_DWORD)v3 != (_DWORD)v4 )
    return 0;
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a1 + 8);
  if ( (a3 & 2) != 0 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
      v7 = **(_QWORD **)(v7 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
      v6 = **(_QWORD **)(v6 + 16);
    if ( v7 == v6 )
      goto LABEL_10;
    return 0;
  }
  if ( v6 != v7 )
    return 0;
LABEL_10:
  if ( (_DWORD)v3 )
  {
    v8 = 32 * v3;
    v9 = a2 - 32 * v4;
    v10 = *(_BYTE *)(a1 + 7) & 0x40;
    v11 = 0;
    v12 = *(_BYTE *)(a2 + 7) & 0x40;
    do
    {
      v13 = a1 - v8;
      if ( (a3 & 2) != 0 )
      {
        if ( v10 )
          v13 = *(_QWORD *)(a1 - 8);
        v7 = *(_QWORD *)(*(_QWORD *)(v13 + v11) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
          v7 = **(_QWORD **)(v7 + 16);
        v14 = v9;
        if ( v12 )
          v14 = *(_QWORD *)(a2 - 8);
        v15 = *(_QWORD *)(*(_QWORD *)(v14 + v11) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
          v15 = **(_QWORD **)(v15 + 16);
        if ( v15 != v7 )
          return 0;
      }
      else
      {
        if ( v10 )
          v13 = *(_QWORD *)(a1 - 8);
        v16 = *(_QWORD *)(*(_QWORD *)(v13 + v11) + 8LL);
        v17 = v9;
        if ( v12 )
          v17 = *(_QWORD *)(a2 - 8);
        v7 = *(_QWORD *)(v17 + v11);
        if ( *(_QWORD *)(v7 + 8) != v16 )
          return 0;
      }
      v11 += 32;
    }
    while ( v8 != v11 );
  }
  return sub_B45D20(a1, a2, a3 & 1, (a3 & 4) != 0, v7);
}
