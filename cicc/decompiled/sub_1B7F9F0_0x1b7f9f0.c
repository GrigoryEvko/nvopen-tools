// Function: sub_1B7F9F0
// Address: 0x1b7f9f0
//
__int64 __fastcall sub_1B7F9F0(_QWORD *a1, __int64 a2, int a3)
{
  int v3; // r15d
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned __int8 v9; // cl
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  int v13; // eax
  unsigned int v14; // esi
  _QWORD **v15; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+18h] [rbp-38h]

  v3 = 1;
  v6 = a2;
  v7 = a1[5];
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v12 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v3 *= (_DWORD)v12;
        continue;
      case 1:
        LODWORD(v8) = 16;
        break;
      case 2:
        LODWORD(v8) = 32;
        break;
      case 3:
      case 9:
        LODWORD(v8) = 64;
        break;
      case 4:
        LODWORD(v8) = 80;
        break;
      case 5:
      case 6:
        LODWORD(v8) = 128;
        break;
      case 7:
        LODWORD(v8) = 8 * sub_15A9520(v7, 0);
        break;
      case 0xB:
        LODWORD(v8) = *(_DWORD *)(a2 + 8) >> 8;
        break;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(v7, a2);
        break;
      case 0xE:
        v16 = *(_QWORD *)(a2 + 24);
        v17 = *(_QWORD *)(a2 + 32);
        v11 = (unsigned int)sub_15A9FE0(v7, v16);
        v8 = 8 * v17 * v11 * ((v11 + ((unsigned __int64)(sub_127FA20(v7, v16) + 7) >> 3) - 1) / v11);
        break;
      case 0xF:
        LODWORD(v8) = 8 * sub_15A9520(v7, *(_DWORD *)(a2 + 8) >> 8);
        break;
    }
    break;
  }
  if ( byte_4FB7CC0 )
    return 0;
  v9 = *(_BYTE *)(v6 + 8);
  if ( v9 == 16 || (unsigned int)v9 - 13 <= 1 )
    return 0;
  v13 = v3 * v8;
  if ( v9 != 11 )
  {
    if ( a3 == 8 && v13 == 16 )
      return sub_16432A0(**(_QWORD ***)(*a1 + 40LL));
    return 0;
  }
  if ( a3 == 8 )
  {
    if ( ((v13 - 8) & 0xFFFFFFF7) != 0 )
      return 0;
    v14 = 2 * v13;
    v15 = *(_QWORD ***)(*a1 + 40LL);
  }
  else
  {
    if ( a3 != 16 || v13 != 8 )
      return 0;
    v14 = 32;
    v15 = *(_QWORD ***)(*a1 + 40LL);
  }
  return sub_1644900(*v15, v14);
}
