// Function: sub_1B7F800
// Address: 0x1b7f800
//
_QWORD *__fastcall sub_1B7F800(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v14; // rax
  __int64 v15; // r13
  unsigned __int64 v16; // r15
  __int64 v17; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)(a3 + 16) != 78 )
    BUG();
  v3 = *(_QWORD *)(a3 - 24);
  if ( *(_BYTE *)(v3 + 16) )
    BUG();
  v6 = *(_DWORD *)(v3 + 36);
  v7 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( v6 == 4085 || v6 == 4057 )
  {
    v9 = *(_QWORD *)a3;
    v8 = *(_QWORD *)(a3 + 24 * (1 - v7));
  }
  else
  {
    v8 = *(_QWORD *)(a3 + 24 * (2 - v7));
    v9 = **(_QWORD **)(a3 + 24 * (1 - v7));
  }
  v10 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v9 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v14 = *(_QWORD *)(v9 + 32);
        v9 = *(_QWORD *)(v9 + 24);
        v10 *= v14;
        continue;
      case 1:
        v11 = 16;
        break;
      case 2:
        v11 = 32;
        break;
      case 3:
      case 9:
        v11 = 64;
        break;
      case 4:
        v11 = 80;
        break;
      case 5:
      case 6:
        v11 = 128;
        break;
      case 7:
        v11 = 8 * (unsigned int)sub_15A9520(a2, 0);
        break;
      case 0xB:
        v11 = *(_DWORD *)(v9 + 8) >> 8;
        break;
      case 0xD:
        v11 = 8LL * *(_QWORD *)sub_15A9930(a2, v9);
        break;
      case 0xE:
        v15 = *(_QWORD *)(v9 + 32);
        v17 = *(_QWORD *)(v9 + 24);
        v16 = (unsigned int)sub_15A9FE0(a2, v17);
        v11 = 8 * v16 * v15 * ((v16 + ((unsigned __int64)(sub_127FA20(a2, v17) + 7) >> 3) - 1) / v16);
        break;
      case 0xF:
        v11 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v9 + 8) >> 8);
        break;
    }
    break;
  }
  v12 = v11 * v10;
  *a1 = v8;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[1] = (unsigned __int64)(v12 + 7) >> 3;
  return a1;
}
