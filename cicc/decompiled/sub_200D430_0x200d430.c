// Function: sub_200D430
// Address: 0x200d430
//
__int64 __fastcall sub_200D430(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  unsigned int v6; // r15d
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // di
  unsigned int v13; // eax
  unsigned __int8 v14; // r8
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rdx
  int v18; // eax
  int v19; // ecx
  bool v20; // al
  unsigned int v21; // eax
  unsigned int v22; // edx
  unsigned int v23; // ecx
  const void **v24; // r8
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 v28; // rdx
  const void **v29; // rdx
  __int64 v30; // rdx
  __int128 v31; // [rsp-10h] [rbp-90h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  _QWORD *v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  unsigned int v35; // [rsp+18h] [rbp-68h]
  unsigned __int8 v37; // [rsp+20h] [rbp-60h]
  unsigned int v38; // [rsp+20h] [rbp-60h]
  const void **v39; // [rsp+20h] [rbp-60h]
  __int64 *v40; // [rsp+28h] [rbp-58h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  char v42[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v43; // [rsp+38h] [rbp-48h]
  __int64 v44; // [rsp+40h] [rbp-40h] BYREF
  __int64 v45; // [rsp+48h] [rbp-38h]

  v34 = 16LL * (unsigned int)a3;
  v10 = *(_QWORD *)(a2 + 40) + v34;
  v11 = *(_QWORD *)(v10 + 8);
  v12 = *(_BYTE *)v10;
  v45 = v11;
  LOBYTE(v44) = v12;
  if ( v12 )
  {
    if ( (unsigned __int8)(v12 - 14) <= 0x5Fu )
    {
      switch ( v12 )
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
          v12 = 3;
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
          v12 = 4;
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
          v12 = 5;
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
          v12 = 6;
          break;
        case 55:
          v12 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v12 = 8;
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
          v12 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v12 = 10;
          break;
        default:
          v12 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  v41 = v11;
  if ( sub_1F58D20((__int64)&v44) )
  {
    v42[0] = sub_1F596B0((__int64)&v44);
    v12 = v42[0];
    v43 = v30;
    if ( v42[0] )
    {
LABEL_3:
      v13 = sub_200D0E0(v12);
      goto LABEL_4;
    }
  }
  else
  {
    v42[0] = 0;
    v43 = v41;
  }
  v13 = sub_1F58D40((__int64)v42);
LABEL_4:
  v40 = *(__int64 **)(a1 + 8);
  v33 = (_QWORD *)v40[6];
  if ( v13 == 32 )
  {
    v14 = 5;
    goto LABEL_8;
  }
  if ( v13 <= 0x20 )
  {
    if ( v13 == 8 )
    {
      v14 = 3;
    }
    else
    {
      v14 = 4;
      if ( v13 != 16 )
      {
        v14 = 2;
        if ( v13 != 1 )
          goto LABEL_22;
      }
    }
LABEL_8:
    v32 = 0;
    goto LABEL_9;
  }
  if ( v13 == 64 )
  {
    v14 = 6;
    goto LABEL_8;
  }
  if ( v13 == 128 )
  {
    v14 = 7;
    goto LABEL_8;
  }
LABEL_22:
  v14 = sub_1F58CC0(v33, v13);
  v32 = v28;
  v40 = *(__int64 **)(a1 + 8);
  v33 = (_QWORD *)v40[6];
LABEL_9:
  v15 = *(_QWORD *)(a2 + 40) + v34;
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  LOBYTE(v44) = v16;
  v45 = v17;
  if ( v16 )
  {
    v19 = word_42FF0A0[(unsigned __int8)(v16 - 14)];
    v20 = (unsigned __int8)(v16 - 56) <= 0x1Du || (unsigned __int8)(v16 - 98) <= 0xBu;
  }
  else
  {
    v37 = v14;
    v18 = sub_1F58D30((__int64)&v44);
    v14 = v37;
    v19 = v18;
    v20 = 0;
  }
  v35 = v19;
  v38 = v14;
  if ( v20 )
  {
    LOBYTE(v21) = sub_1D154A0(v14, v19);
    v22 = v38;
    v23 = v35;
    v24 = 0;
    if ( (_BYTE)v21 )
      goto LABEL_13;
  }
  else
  {
    LOBYTE(v21) = sub_1D15020(v14, v19);
    v23 = v35;
    v22 = v38;
    v24 = 0;
    if ( (_BYTE)v21 )
      goto LABEL_13;
  }
  v21 = sub_1F593D0(v33, v22, v32, v23);
  v6 = v21;
  v24 = v29;
LABEL_13:
  v25 = *(_QWORD *)(a2 + 72);
  LOBYTE(v6) = v21;
  v44 = v25;
  if ( v25 )
  {
    v39 = v24;
    sub_1623A60((__int64)&v44, v25, 2);
    v24 = v39;
  }
  *((_QWORD *)&v31 + 1) = a3;
  *(_QWORD *)&v31 = a2;
  LODWORD(v45) = *(_DWORD *)(a2 + 64);
  v26 = sub_1D309E0(v40, 158, (__int64)&v44, v6, v24, 0, a4, a5, a6, v31);
  if ( v44 )
    sub_161E7C0((__int64)&v44, v44);
  return v26;
}
