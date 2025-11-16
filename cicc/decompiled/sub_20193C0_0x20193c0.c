// Function: sub_20193C0
// Address: 0x20193c0
//
__int64 __fastcall sub_20193C0(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r12
  char v5; // r13
  int v6; // r8d
  int v7; // r9d
  int v8; // ebx
  int v9; // r12d
  __int64 result; // rax
  unsigned int v11; // r15d
  char v13; // al
  __int64 v14; // rdx
  unsigned int v15; // [rsp+8h] [rbp-58h]
  unsigned int v16; // [rsp+Ch] [rbp-54h]
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  char v19[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v20; // [rsp+28h] [rbp-38h]

  v4 = a1;
  v17 = a1;
  v18 = a2;
  if ( (_BYTE)a1 )
  {
    if ( (unsigned __int8)(a1 - 14) <= 0x5Fu )
    {
      switch ( (char)a1 )
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
          v5 = a1;
          v4 = 3;
          goto LABEL_16;
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
          v5 = a1;
          v4 = 4;
          goto LABEL_16;
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
          v5 = a1;
          v4 = 5;
          goto LABEL_16;
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
          v5 = a1;
          v4 = 6;
          goto LABEL_16;
        case 55:
          v5 = a1;
          v16 = (unsigned int)sub_2018C90(7) >> 3;
          goto LABEL_17;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v5 = a1;
          v4 = 8;
          goto LABEL_16;
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
          v5 = a1;
          v4 = 9;
          goto LABEL_16;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v5 = a1;
          v4 = 10;
          goto LABEL_16;
        default:
          v5 = a1;
          v4 = 2;
          goto LABEL_16;
      }
    }
LABEL_3:
    v19[0] = a1;
    v5 = a1;
    v20 = v18;
    if ( !(_BYTE)a1 )
      goto LABEL_4;
LABEL_16:
    v16 = (unsigned int)sub_2018C90(v4) >> 3;
    if ( !v5 )
      goto LABEL_5;
    goto LABEL_17;
  }
  if ( !sub_1F58D20((__int64)&v17) )
    goto LABEL_3;
  v13 = sub_1F596B0((__int64)&v17);
  v5 = v17;
  v4 = v13;
  v20 = v14;
  v19[0] = v13;
  if ( v13 )
    goto LABEL_16;
LABEL_4:
  v16 = (unsigned int)sub_1F58D40((__int64)v19) >> 3;
  if ( !v5 )
  {
LABEL_5:
    v15 = sub_1F58D30((__int64)&v17);
    goto LABEL_6;
  }
LABEL_17:
  v15 = word_4301260[(unsigned __int8)(v5 - 14)];
LABEL_6:
  v8 = 0;
  v9 = 0;
  result = v15;
  if ( v15 )
  {
    do
    {
      if ( v16 )
      {
        result = *(unsigned int *)(a3 + 8);
        v11 = v16 - 1;
        do
        {
          if ( *(_DWORD *)(a3 + 12) <= (unsigned int)result )
          {
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v6, v7);
            result = *(unsigned int *)(a3 + 8);
          }
          *(_DWORD *)(*(_QWORD *)a3 + 4 * result) = v11 + v8;
          result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
          *(_DWORD *)(a3 + 8) = result;
        }
        while ( v11-- != 0 );
      }
      ++v9;
      v8 += v16;
    }
    while ( v15 != v9 );
  }
  return result;
}
