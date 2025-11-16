// Function: sub_1D169E0
// Address: 0x1d169e0
//
__int64 __fastcall sub_1D169E0(_QWORD *a1, _QWORD *a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r13
  _QWORD *v5; // rbx
  int v6; // eax
  bool v7; // zf
  unsigned int v8; // r12d
  __int64 v10; // rsi
  char v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // r8
  __int64 v16; // rax
  int v17; // edx
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  _QWORD *v23; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v24[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = a3;
  v5 = a1;
  v6 = *((unsigned __int16 *)a1 + 12);
  LOBYTE(a3) = v6 == 32;
  LOBYTE(a4) = v6 == 32 || v6 == 10;
  if ( (_BYTE)a4 )
  {
    v7 = *(_QWORD *)(v4 + 16) == 0;
    v23 = a1;
    if ( v7 )
LABEL_31:
      sub_4263D6(a1, a2, a3);
    return (*(unsigned int (__fastcall **)(__int64, _QWORD **))(v4 + 24))(v4, &v23);
  }
  else
  {
    v8 = a4;
    if ( v6 == 104 )
    {
      v10 = a1[5] + 16LL * (unsigned int)a2;
      v11 = *(_BYTE *)v10;
      v12 = *(_QWORD *)(v10 + 8);
      LOBYTE(v24[0]) = v11;
      v24[1] = v12;
      if ( v11 )
      {
        if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
        {
          switch ( v11 )
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
              v11 = 3;
              v12 = 0;
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
              v11 = 4;
              v12 = 0;
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
              v11 = 5;
              v12 = 0;
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
              v11 = 6;
              v12 = 0;
              break;
            case 55:
              v11 = 7;
              v12 = 0;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v11 = 8;
              v12 = 0;
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
              v11 = 9;
              v12 = 0;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v11 = 10;
              v12 = 0;
              break;
            default:
              v11 = 2;
              v12 = 0;
              break;
          }
        }
      }
      else
      {
        v22 = v12;
        a1 = v24;
        v18 = sub_1F58D20(v24);
        v12 = v22;
        if ( v18 )
        {
          a1 = v24;
          v11 = sub_1F596B0(v24);
          v12 = v19;
        }
      }
      v13 = *((unsigned int *)v5 + 14);
      if ( (_DWORD)v13 )
      {
        v14 = 0;
        a2 = v24;
        v15 = 40 * v13;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v5[4] + v14);
          v17 = *(unsigned __int16 *)(v16 + 24);
          if ( v17 != 10 && v17 != 32 )
            break;
          a3 = *(_QWORD *)(v16 + 40);
          if ( v11 != *(_BYTE *)a3 || *(_QWORD *)(a3 + 8) != v12 && !v11 )
            break;
          v7 = *(_QWORD *)(v4 + 16) == 0;
          v20 = v15;
          v21 = v12;
          v24[0] = *(_QWORD *)(v5[4] + v14);
          if ( v7 )
            goto LABEL_31;
          a1 = (_QWORD *)v4;
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(v4 + 24))(v4) )
            break;
          v15 = v20;
          v14 += 40;
          v12 = v21;
          if ( v20 == v14 )
            return 1;
        }
      }
      else
      {
        return 1;
      }
    }
  }
  return v8;
}
