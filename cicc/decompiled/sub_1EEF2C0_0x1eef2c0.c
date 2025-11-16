// Function: sub_1EEF2C0
// Address: 0x1eef2c0
//
unsigned __int64 __fastcall sub_1EEF2C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+20h] [rbp-40h]

  v2 = 1;
  v3 = *(_QWORD *)(a2 + 56);
  v4 = (unsigned int)sub_15A9FE0(a1, v3);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v3 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v12 = *(_QWORD *)(v3 + 32);
        v3 = *(_QWORD *)(v3 + 24);
        v2 *= v12;
        continue;
      case 1:
        v5 = 16;
        goto LABEL_4;
      case 2:
        v5 = 32;
        goto LABEL_4;
      case 3:
      case 9:
        v5 = 64;
        goto LABEL_4;
      case 4:
        v5 = 80;
        goto LABEL_4;
      case 5:
      case 6:
        v5 = 128;
        goto LABEL_4;
      case 7:
        v5 = 8 * (unsigned int)sub_15A9520(a1, 0);
        goto LABEL_4;
      case 0xB:
        v5 = *(_DWORD *)(v3 + 8) >> 8;
        goto LABEL_4;
      case 0xD:
        v5 = 8LL * *(_QWORD *)sub_15A9930(a1, v3);
        goto LABEL_4;
      case 0xE:
        v15 = *(_QWORD *)(v3 + 24);
        sub_15A9FE0(a1, v15);
        v10 = v15;
        v11 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v10 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v13 = *(_QWORD *)(v10 + 32);
              v10 = *(_QWORD *)(v10 + 24);
              v11 *= v13;
              continue;
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 9:
            case 0xB:
              goto LABEL_23;
            case 7:
              sub_15A9520(a1, 0);
              goto LABEL_23;
            case 0xD:
              sub_15A9930(a1, v10);
              goto LABEL_23;
            case 0xE:
              v14 = *(_QWORD *)(v10 + 24);
              sub_15A9FE0(a1, v14);
              sub_127FA20(a1, v14);
              goto LABEL_23;
            case 0xF:
              sub_15A9520(a1, *(_DWORD *)(v10 + 8) >> 8);
LABEL_23:
              JUMPOUT(0x1EEF4A6);
          }
        }
      case 0xF:
        v5 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v3 + 8) >> 8);
LABEL_4:
        v6 = (v4 + ((unsigned __int64)(v5 * v2 + 7) >> 3) - 1) / v4 * v4;
        if ( (unsigned __int8)sub_15F8BF0(a2) )
        {
          v7 = *(_QWORD *)(a2 - 24);
          if ( *(_BYTE *)(v7 + 16) == 13 )
          {
            if ( *(_DWORD *)(v7 + 32) <= 0x40u )
              v8 = *(_QWORD *)(v7 + 24);
            else
              v8 = **(_QWORD **)(v7 + 24);
            v6 *= v8;
          }
          else
          {
            return 0;
          }
        }
        return v6;
    }
  }
}
