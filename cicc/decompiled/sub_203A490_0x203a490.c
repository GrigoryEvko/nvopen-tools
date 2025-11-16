// Function: sub_203A490
// Address: 0x203a490
//
__int64 __fastcall sub_203A490(__int64 *a1, unsigned __int64 a2, __m128i a3, double a4, __m128i a5)
{
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r13
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int128 v13; // rax
  __int64 v14; // rsi
  int v15; // ecx
  char v16; // al
  unsigned __int8 v17; // al
  __int64 v18; // rdx
  unsigned int v19; // eax
  const void **v20; // r8
  unsigned int v21; // edi
  __int64 *v22; // rax
  unsigned int v23; // edx
  __int64 v24; // r13
  const __m128i *v25; // r9
  const void **v27; // rdx
  __int128 v28; // [rsp-50h] [rbp-110h]
  unsigned int v29; // [rsp+8h] [rbp-B8h]
  __int64 v30; // [rsp+10h] [rbp-B0h]
  _QWORD *v31; // [rsp+18h] [rbp-A8h]
  __int128 v32; // [rsp+20h] [rbp-A0h]
  int v33; // [rsp+30h] [rbp-90h]
  unsigned int v34; // [rsp+30h] [rbp-90h]
  unsigned int v35; // [rsp+38h] [rbp-88h]
  char v36; // [rsp+3Ch] [rbp-84h]
  unsigned int v37; // [rsp+50h] [rbp-70h] BYREF
  __int64 v38; // [rsp+58h] [rbp-68h]
  _BYTE v39[8]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+68h] [rbp-58h]
  __int64 v41; // [rsp+70h] [rbp-50h] BYREF
  int v42; // [rsp+78h] [rbp-48h]
  __int64 v43; // [rsp+80h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v41,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v7 = *(_QWORD **)(a2 + 32);
  v8 = v7[10];
  v9 = v8;
  LOBYTE(v37) = v42;
  v10 = v7[11];
  v38 = v43;
  v11 = *(_QWORD *)(v8 + 40) + 16LL * *((unsigned int *)v7 + 22);
  LOBYTE(v8) = *(_BYTE *)v11;
  v12 = *(_QWORD *)(v11 + 8);
  v39[0] = v8;
  v40 = v12;
  *(_QWORD *)&v13 = sub_20363F0((__int64)a1, v7[15], v7[16]);
  v14 = *(_QWORD *)(a2 + 72);
  v32 = v13;
  LOBYTE(v13) = *(_BYTE *)(a2 + 27);
  v41 = v14;
  v36 = ((unsigned __int8)v13 >> 2) & 3;
  if ( v14 )
    sub_1623A60((__int64)&v41, v14, 2);
  v42 = *(_DWORD *)(a2 + 64);
  if ( !(_BYTE)v37 )
  {
    v15 = sub_1F58D30((__int64)&v37);
    v16 = v39[0];
    if ( v39[0] )
      goto LABEL_5;
LABEL_7:
    v33 = v15;
    v17 = sub_1F596B0((__int64)v39);
    v15 = v33;
    v30 = v18;
    goto LABEL_8;
  }
  v15 = word_4305480[(unsigned __int8)(v37 - 14)];
  v16 = v39[0];
  if ( !v39[0] )
    goto LABEL_7;
LABEL_5:
  switch ( v16 )
  {
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 56:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
      v17 = 2;
      break;
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
      v17 = 3;
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
      v17 = 4;
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
      v17 = 5;
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
      v17 = 6;
      break;
    case 55:
      v17 = 7;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      v17 = 8;
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
      v17 = 9;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      v17 = 10;
      break;
  }
  v30 = 0;
LABEL_8:
  v35 = v15;
  v31 = *(_QWORD **)(a1[1] + 48);
  v34 = v17;
  LOBYTE(v19) = sub_1D15020(v17, v15);
  v20 = 0;
  if ( !(_BYTE)v19 )
  {
    v19 = sub_1F593D0(v31, v34, v30, v35);
    v29 = v19;
    v20 = v27;
  }
  v21 = v29;
  LOBYTE(v21) = v19;
  v22 = sub_2030300(a1, v9, v10, v21, v20, 1, a3, a4, a5);
  *((_QWORD *)&v28 + 1) = v23 | v10 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v28 = v22;
  v24 = sub_1D257D0(
          (_QWORD *)a1[1],
          v37,
          v38,
          (__int64)&v41,
          **(_QWORD **)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          v28,
          v32,
          *(unsigned __int8 *)(a2 + 88),
          *(_QWORD *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          v36,
          (*(_BYTE *)(a2 + 27) & 0x10) != 0);
  sub_2013400((__int64)a1, a2, 1, v24, (__m128i *)1, v25);
  if ( v41 )
    sub_161E7C0((__int64)&v41, v41);
  return v24;
}
