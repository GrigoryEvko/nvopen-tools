// Function: sub_12BE0A0
// Address: 0x12be0a0
//
unsigned __int64 __fastcall sub_12BE0A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rax
  __int64 v7; // r15
  unsigned int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rax
  int v16; // eax
  _QWORD *v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v2 = a2;
  v3 = 1;
  v4 = (unsigned int)sub_15A9FE0(a1, a2);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v12 = *(_QWORD *)(v2 + 32);
        v2 = *(_QWORD *)(v2 + 24);
        v3 *= v12;
        continue;
      case 1:
        v5 = 16;
        break;
      case 2:
        v5 = 32;
        break;
      case 3:
      case 9:
        v5 = 64;
        break;
      case 4:
        v5 = 80;
        break;
      case 5:
      case 6:
        v5 = 128;
        break;
      case 7:
        v5 = 8 * (unsigned int)sub_15A9520(a1, 0);
        break;
      case 0xB:
        v5 = *(_DWORD *)(v2 + 8) >> 8;
        break;
      case 0xD:
        v5 = 8LL * *(_QWORD *)sub_15A9930(a1, v2);
        break;
      case 0xE:
        v7 = *(_QWORD *)(v2 + 32);
        v18 = *(_QWORD *)(v2 + 24);
        v8 = sub_15A9FE0(a1, v18);
        v9 = v18;
        v10 = 1;
        v11 = v8;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v9 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v15 = *(_QWORD *)(v9 + 32);
              v9 = *(_QWORD *)(v9 + 24);
              v10 *= v15;
              continue;
            case 1:
              v13 = 16;
              goto LABEL_17;
            case 2:
              v13 = 32;
              goto LABEL_17;
            case 3:
            case 9:
              v13 = 64;
              goto LABEL_17;
            case 4:
              v13 = 80;
              goto LABEL_17;
            case 5:
            case 6:
              v13 = 128;
              goto LABEL_17;
            case 7:
              v19 = v10;
              v14 = sub_15A9520(a1, 0);
              v10 = v19;
              v13 = (unsigned int)(8 * v14);
              goto LABEL_17;
            case 0xB:
              v13 = *(_DWORD *)(v9 + 8) >> 8;
              goto LABEL_17;
            case 0xD:
              v21 = v10;
              v17 = (_QWORD *)sub_15A9930(a1, v9);
              v10 = v21;
              v13 = 8LL * *v17;
              goto LABEL_17;
            case 0xE:
              JUMPOUT(0x12BE283);
            case 0xF:
              v20 = v10;
              v16 = sub_15A9520(a1, *(_DWORD *)(v9 + 8) >> 8);
              v10 = v20;
              v13 = (unsigned int)(8 * v16);
LABEL_17:
              v5 = 8 * v11 * v7 * ((v11 + ((unsigned __int64)(v13 * v10 + 7) >> 3) - 1) / v11);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v5 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v2 + 8) >> 8);
        break;
    }
    return v4 * ((v4 + ((unsigned __int64)(v5 * v3 + 7) >> 3) - 1) / v4);
  }
}
