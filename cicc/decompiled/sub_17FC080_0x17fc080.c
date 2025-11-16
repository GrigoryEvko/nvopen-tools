// Function: sub_17FC080
// Address: 0x17fc080
//
__int64 __fastcall sub_17FC080(__int64 a1, __int64 a2)
{
  int v3; // ebx
  __int64 v4; // rsi
  __int64 v5; // rax
  unsigned int v6; // eax
  bool v9; // cc
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // r14
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v3 = 1;
  v4 = *(_QWORD *)(a1 + 24);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v4 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v14 = *(_QWORD *)(v4 + 32);
        v4 = *(_QWORD *)(v4 + 24);
        v3 *= (_DWORD)v14;
        continue;
      case 1:
        LODWORD(v5) = 16;
        break;
      case 2:
        LODWORD(v5) = 32;
        break;
      case 3:
      case 9:
        LODWORD(v5) = 64;
        break;
      case 4:
        LODWORD(v5) = 80;
        break;
      case 5:
      case 6:
        LODWORD(v5) = 128;
        break;
      case 7:
        LODWORD(v5) = 8 * sub_15A9520(a2, 0);
        break;
      case 0xB:
        LODWORD(v5) = *(_DWORD *)(v4 + 8) >> 8;
        break;
      case 0xD:
        v5 = 8LL * *(_QWORD *)sub_15A9930(a2, v4);
        break;
      case 0xE:
        v11 = *(_QWORD *)(v4 + 24);
        v12 = 1;
        v20 = *(_QWORD *)(v4 + 32);
        v13 = (unsigned int)sub_15A9FE0(a2, v11);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v11 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v16 = *(_QWORD *)(v11 + 32);
              v11 = *(_QWORD *)(v11 + 24);
              v12 *= v16;
              continue;
            case 1:
              v15 = 16;
              break;
            case 2:
              v15 = 32;
              break;
            case 3:
            case 9:
              v15 = 64;
              break;
            case 4:
              v15 = 80;
              break;
            case 5:
            case 6:
              v15 = 128;
              break;
            case 7:
              v15 = 8 * (unsigned int)sub_15A9520(a2, 0);
              break;
            case 0xB:
              v15 = *(_DWORD *)(v11 + 8) >> 8;
              break;
            case 0xD:
              v15 = 8LL * *(_QWORD *)sub_15A9930(a2, v11);
              break;
            case 0xE:
              v18 = *(_QWORD *)(v11 + 24);
              v19 = *(_QWORD *)(v11 + 32);
              v17 = (unsigned int)sub_15A9FE0(a2, v18);
              v15 = 8 * v17 * v19 * ((v17 + ((unsigned __int64)(sub_127FA20(a2, v18) + 7) >> 3) - 1) / v17);
              break;
            case 0xF:
              v15 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v11 + 8) >> 8);
              break;
          }
          break;
        }
        v5 = 8 * v13 * v20 * ((v13 + ((unsigned __int64)(v15 * v12 + 7) >> 3) - 1) / v13);
        break;
      case 0xF:
        LODWORD(v5) = 8 * sub_15A9520(a2, *(_DWORD *)(v4 + 8) >> 8);
        break;
    }
    break;
  }
  v6 = (v5 * v3 + 7) & 0xFFFFFFF8;
  if ( ((v6 - 8) & 0xFFFFFFF0) != 0 && ((v6 - 32) & 0xFFFFFFD8) != 0 && v6 != 128 )
    return 0xFFFFFFFFLL;
  _EDX = v6 >> 3;
  __asm { tzcnt   edx, edx }
  v9 = v6 <= 7;
  result = 32;
  if ( !v9 )
    return _EDX;
  return result;
}
