// Function: sub_1D39370
// Address: 0x1d39370
//
__int64 __fastcall sub_1D39370(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        const void **a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v13; // rdi
  char v14; // r15
  __int64 v15; // rdx
  int v16; // eax
  unsigned int v17; // edx
  int v18; // r15d
  unsigned int v19; // esi
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rax
  int v25; // eax
  char v26[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v27; // [rsp+18h] [rbp-38h]

  v13 = *a1;
  v14 = *(_BYTE *)v13;
  if ( *(_BYTE *)v13 )
  {
    if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
    {
      switch ( v14 )
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
          v14 = 3;
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
          v14 = 4;
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
          v14 = 5;
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
          v14 = 6;
          break;
        case 55:
          v14 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v14 = 8;
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
          v14 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v14 = 10;
          break;
        default:
          v14 = 2;
          break;
      }
      goto LABEL_13;
    }
    goto LABEL_3;
  }
  if ( !(unsigned __int8)sub_1F58D20(v13) )
  {
LABEL_3:
    v15 = *(_QWORD *)(v13 + 8);
    goto LABEL_4;
  }
  v14 = sub_1F596B0(v13);
LABEL_4:
  v26[0] = v14;
  v27 = v15;
  if ( !v14 )
  {
    v16 = sub_1F58D40(v26, a2, v15, a4, a5, a6);
    v17 = *(_DWORD *)(a2 + 8);
    v18 = v16;
    v19 = v17 - v16;
    if ( v17 <= 0x40 )
      goto LABEL_6;
    goto LABEL_14;
  }
LABEL_13:
  v25 = sub_1D13440(v14);
  v17 = *(_DWORD *)(a2 + 8);
  v18 = v25;
  v19 = v17 - v25;
  if ( v17 <= 0x40 )
  {
LABEL_6:
    v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
    v21 = 0;
    if ( v19 != v17 )
      v21 = (__int64)((v20 & (*(_QWORD *)a2 << v19)) << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
    goto LABEL_8;
  }
LABEL_14:
  sub_16A7DC0((__int64 *)a2, v19);
  v17 = *(_DWORD *)(a2 + 8);
  v19 = v17 - v18;
  if ( v17 > 0x40 )
  {
    sub_16A5E70(a2, v19);
    return sub_1D38970(a1[2], a2, a1[1], a3, a4, 0, a7, a8, a9, 0);
  }
  v21 = (__int64)(*(_QWORD *)a2 << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
  v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
LABEL_8:
  v22 = v21 >> 63;
  v23 = v21 >> v19;
  if ( v17 == v19 )
    v23 = v22;
  *(_QWORD *)a2 = v20 & v23;
  return sub_1D38970(a1[2], a2, a1[1], a3, a4, 0, a7, a8, a9, 0);
}
