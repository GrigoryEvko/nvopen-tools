// Function: sub_2129220
// Address: 0x2129220
//
__int64 *__fastcall sub_2129220(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  char v9; // dl
  __int16 v10; // ax
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 *result; // rax
  unsigned __int8 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  __int64 v19; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v8 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL) + 16LL * *(unsigned int *)(v5 + 48);
  v9 = *(_BYTE *)v8;
  v19 = *(_QWORD *)(v8 + 8);
  v10 = *(_WORD *)(a2 + 24);
  LOBYTE(v18) = v9;
  if ( v10 != 135 )
  {
LABEL_2:
    if ( v10 != 134 )
    {
      v11 = v18;
      v12 = v19;
LABEL_4:
      v17 = sub_200E230((_QWORD *)a1, v6, v7, v11, v12, a3, a4, a5);
      return sub_1D2E2F0(
               *(_QWORD **)(a1 + 8),
               (__int64 *)a2,
               v17,
               v7 & 0xFFFFFFFF00000000LL | v13,
               *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
               *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
               *(_OWORD *)(*(_QWORD *)(a2 + 32) + 80LL));
    }
    v15 = v18;
    if ( (_BYTE)v18 )
    {
      if ( (unsigned __int8)(v18 - 14) <= 0x5Fu )
      {
        switch ( (char)v18 )
        {
          case 24:
          case 25:
          case 26:
          case 27:
          case 28:
          case 29:
          case 30:
          case 31:
          case 32:
          case 62:
          case 63:
          case 64:
          case 65:
          case 66:
          case 67:
            v15 = 3;
            break;
          case 33:
          case 34:
          case 35:
          case 36:
          case 37:
          case 38:
          case 39:
          case 40:
          case 68:
          case 69:
          case 70:
          case 71:
          case 72:
          case 73:
            v15 = 4;
            break;
          case 41:
          case 42:
          case 43:
          case 44:
          case 45:
          case 46:
          case 47:
          case 48:
          case 74:
          case 75:
          case 76:
          case 77:
          case 78:
          case 79:
            v15 = 5;
            break;
          case 49:
          case 50:
          case 51:
          case 52:
          case 53:
          case 54:
          case 80:
          case 81:
          case 82:
          case 83:
          case 84:
          case 85:
            v15 = 6;
            break;
          case 55:
            v15 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v15 = 8;
            break;
          case 89:
          case 90:
          case 91:
          case 92:
          case 93:
          case 101:
          case 102:
          case 103:
          case 104:
          case 105:
            v15 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v15 = 10;
            break;
          default:
            v15 = 2;
            break;
        }
        v12 = 0;
        goto LABEL_11;
      }
    }
    else if ( sub_1F58D20((__int64)&v18) )
    {
      v15 = sub_1F596B0((__int64)&v18);
      v12 = v16;
      goto LABEL_11;
    }
    v12 = v19;
LABEL_11:
    v11 = v15;
    goto LABEL_4;
  }
  result = sub_203AD40((__int64 *)a1, a2);
  if ( !result )
  {
    v10 = *(_WORD *)(a2 + 24);
    goto LABEL_2;
  }
  return result;
}
