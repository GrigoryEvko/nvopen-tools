// Function: sub_127FA20
// Address: 0x127fa20
//
__int64 __fastcall sub_127FA20(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // r13
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // r14
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+18h] [rbp-38h]

  v2 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v9 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v2 *= v9;
        continue;
      case 1:
        v3 = 16;
        break;
      case 2:
        v3 = 32;
        break;
      case 3:
      case 9:
        v3 = 64;
        break;
      case 4:
        v3 = 80;
        break;
      case 5:
      case 6:
        v3 = 128;
        break;
      case 7:
        v3 = 8 * (unsigned int)sub_15A9520(a1, 0);
        break;
      case 0xB:
        v3 = *(_DWORD *)(a2 + 8) >> 8;
        break;
      case 0xD:
        v3 = 8LL * *(_QWORD *)sub_15A9930(a1, a2);
        break;
      case 0xE:
        v5 = *(_QWORD *)(a2 + 24);
        v6 = *(_QWORD *)(a2 + 32);
        v7 = 1;
        v8 = (unsigned int)sub_15A9FE0(a1, v5);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v5 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v11 = *(_QWORD *)(v5 + 32);
              v5 = *(_QWORD *)(v5 + 24);
              v7 *= v11;
              continue;
            case 1:
              v10 = 16;
              break;
            case 2:
              v10 = 32;
              break;
            case 3:
            case 9:
              v10 = 64;
              break;
            case 4:
              v10 = 80;
              break;
            case 5:
            case 6:
              v10 = 128;
              break;
            case 7:
              v10 = 8 * (unsigned int)sub_15A9520(a1, 0);
              break;
            case 0xB:
              v10 = *(_DWORD *)(v5 + 8) >> 8;
              break;
            case 0xD:
              v10 = 8LL * *(_QWORD *)sub_15A9930(a1, v5);
              break;
            case 0xE:
              v13 = *(_QWORD *)(v5 + 24);
              v14 = *(_QWORD *)(v5 + 32);
              v12 = (unsigned int)sub_15A9FE0(a1, v13);
              v10 = 8 * v14 * v12 * ((v12 + ((unsigned __int64)(sub_127FA20(a1, v13) + 7) >> 3) - 1) / v12);
              break;
            case 0xF:
              v10 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v5 + 8) >> 8);
              break;
          }
          break;
        }
        v3 = 8 * v8 * v6 * ((v8 + ((unsigned __int64)(v10 * v7 + 7) >> 3) - 1) / v8);
        break;
      case 0xF:
        v3 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(a2 + 8) >> 8);
        break;
    }
    return v2 * v3;
  }
}
