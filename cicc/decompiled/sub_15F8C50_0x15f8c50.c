// Function: sub_15F8C50
// Address: 0x15f8c50
//
__int64 __fastcall sub_15F8C50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // r15
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-48h]

  v3 = 1;
  v5 = *(_QWORD *)(a2 + 56);
  v7 = (unsigned int)sub_15A9FE0(a3, v5);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v5 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v15 = *(_QWORD *)(v5 + 32);
        v5 = *(_QWORD *)(v5 + 24);
        v3 *= v15;
        continue;
      case 1:
        v8 = 16;
        goto LABEL_4;
      case 2:
        v8 = 32;
        goto LABEL_4;
      case 3:
      case 9:
        v8 = 64;
        goto LABEL_4;
      case 4:
        v8 = 80;
        goto LABEL_4;
      case 5:
      case 6:
        v8 = 128;
        goto LABEL_4;
      case 7:
        v8 = 8 * (unsigned int)sub_15A9520(a3, 0);
        goto LABEL_4;
      case 0xB:
        v8 = *(_DWORD *)(v5 + 8) >> 8;
        goto LABEL_4;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(a3, v5);
        goto LABEL_4;
      case 0xE:
        v17 = *(_QWORD *)(v5 + 24);
        sub_15A9FE0(a3, v17);
        v13 = v17;
        v14 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v13 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v16 = *(_QWORD *)(v13 + 32);
              v13 = *(_QWORD *)(v13 + 24);
              v14 *= v16;
              continue;
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 9:
            case 0xB:
              goto LABEL_24;
            case 7:
              sub_15A9520(a3, 0);
              goto LABEL_24;
            case 0xD:
              sub_15A9930(a3, v13);
              goto LABEL_24;
            case 0xE:
              sub_12BE0A0(a3, *(_QWORD *)(v13 + 24));
LABEL_24:
              JUMPOUT(0x15F8E3B);
            case 0xF:
              sub_15A9520(a3, *(_DWORD *)(v13 + 8) >> 8);
              JUMPOUT(0x15F8EA8);
          }
        }
      case 0xF:
        v8 = 8 * (unsigned int)sub_15A9520(a3, *(_DWORD *)(v5 + 8) >> 8);
LABEL_4:
        v9 = 8 * v7 * ((v7 + ((unsigned __int64)(v8 * v3 + 7) >> 3) - 1) / v7);
        if ( !(unsigned __int8)sub_15F8BF0(a2) )
          goto LABEL_9;
        v10 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v10 + 16) == 13 )
        {
          if ( *(_DWORD *)(v10 + 32) <= 0x40u )
            v11 = *(_QWORD *)(v10 + 24);
          else
            v11 = **(_QWORD **)(v10 + 24);
          v9 *= v11;
LABEL_9:
          *(_BYTE *)(a1 + 8) = 1;
          *(_QWORD *)a1 = v9;
        }
        else
        {
          *(_BYTE *)(a1 + 8) = 0;
        }
        return a1;
    }
  }
}
