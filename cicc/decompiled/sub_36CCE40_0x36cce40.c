// Function: sub_36CCE40
// Address: 0x36cce40
//
__int64 __fastcall sub_36CCE40(__int64 a1, unsigned __int8 a2, unsigned int a3)
{
  int v3; // r15d
  unsigned int v4; // r12d
  __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r9
  __int16 v9; // ax
  unsigned __int8 v10; // al
  int v11; // eax
  char v13; // al
  int v14; // ecx
  int v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  char v17; // [rsp+8h] [rbp-48h]
  __int64 i; // [rsp+10h] [rbp-40h]
  unsigned __int8 v19; // [rsp+1Dh] [rbp-33h]
  char v20; // [rsp+1Eh] [rbp-32h]

  v4 = a3;
  v20 = a3;
  LOBYTE(v4) = a2 | a3;
  if ( a2 | (unsigned __int8)a3 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    v4 = 0;
    for ( i = a1 + 312; a1 + 8 != v5; v5 = *(_QWORD *)(v5 + 8) )
    {
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)(v5 - 24) & 0xFu) - 4 <= 1 )
        continue;
      if ( sub_B2FC80(v5 - 56) )
        continue;
      v6 = *(_QWORD *)(v5 - 32);
      if ( *(_BYTE *)(v6 + 8) != 16 )
        continue;
      v7 = *(_QWORD *)(v5 - 48);
      if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
        v7 = **(_QWORD **)(v7 + 16);
      v19 = v20 & (*(_DWORD *)(v7 + 8) >> 8 == 1) | a2 & (*(_DWORD *)(v7 + 8) >> 8 == 3);
      if ( !v19 )
        continue;
      v8 = *(_QWORD *)(v6 + 32);
      if ( !v8 )
        continue;
      v9 = (*(_WORD *)(v5 - 22) >> 1) & 0x3F;
      if ( v9 )
      {
        v17 = v9 - 1;
        if ( (unsigned __int8)(v9 - 1) > 3u )
          continue;
        v13 = sub_36CCDB0(i, v8, *(_QWORD *)(v6 + 24));
        LOBYTE(v3) = v13;
        v14 = v3;
        BYTE1(v14) = 1;
        v3 = v14;
        if ( v17 == v13 )
          continue;
      }
      else
      {
        v15 = *(_QWORD *)(v6 + 32);
        v16 = *(_QWORD *)(v6 + 24);
        v10 = sub_AE5260(i, v16);
        if ( v10 <= 3u )
          v10 = sub_36CCDB0(i, v15, v16);
        LOBYTE(v3) = v10;
        v11 = v3;
        BYTE1(v11) = 1;
        v3 = v11;
      }
      sub_B2F740(v5 - 56, v3);
      v4 = v19;
    }
  }
  return v4;
}
