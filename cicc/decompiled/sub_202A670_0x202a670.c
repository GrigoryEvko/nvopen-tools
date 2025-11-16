// Function: sub_202A670
// Address: 0x202a670
//
__int64 *__fastcall sub_202A670(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r13d
  char *v8; // rax
  __int64 v9; // rsi
  char v10; // dl
  const void **v11; // rax
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rax
  int v15; // ecx
  char v16; // al
  unsigned __int8 v17; // al
  __int64 v18; // rdx
  unsigned int v19; // r15d
  unsigned int v20; // eax
  const void **v21; // r8
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // rsi
  int v25; // edx
  __int64 v26; // rax
  __int64 *v27; // rdi
  int v28; // edx
  __int64 *v29; // r14
  const void **v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-B8h]
  _QWORD *v33; // [rsp+10h] [rbp-B0h]
  int v34; // [rsp+18h] [rbp-A8h]
  unsigned int v35; // [rsp+18h] [rbp-A8h]
  const void **v36; // [rsp+18h] [rbp-A8h]
  unsigned int v37; // [rsp+40h] [rbp-80h] BYREF
  const void **v38; // [rsp+48h] [rbp-78h]
  __int128 v39; // [rsp+50h] [rbp-70h] BYREF
  __int128 v40; // [rsp+60h] [rbp-60h] BYREF
  __int64 v41; // [rsp+70h] [rbp-50h] BYREF
  int v42; // [rsp+78h] [rbp-48h]
  _BYTE v43[8]; // [rsp+80h] [rbp-40h] BYREF
  __int64 v44; // [rsp+88h] [rbp-38h]

  v8 = *(char **)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *v8;
  v11 = (const void **)*((_QWORD *)v8 + 1);
  *(_QWORD *)&v39 = 0;
  DWORD2(v39) = 0;
  LOBYTE(v37) = v10;
  v38 = v11;
  *(_QWORD *)&v40 = 0;
  DWORD2(v40) = 0;
  v41 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v41, v9, 2);
  v42 = *(_DWORD *)(a2 + 64);
  sub_2017DE0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), &v39, &v40);
  v12 = *(_QWORD *)(v39 + 40) + 16LL * DWORD2(v39);
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v43[0] = v13;
  v44 = v14;
  if ( !v13 )
  {
    v15 = sub_1F58D30((__int64)v43);
    v16 = v37;
    if ( (_BYTE)v37 )
      goto LABEL_5;
LABEL_7:
    v34 = v15;
    v17 = sub_1F596B0((__int64)&v37);
    v15 = v34;
    v32 = v18;
    goto LABEL_8;
  }
  v15 = word_4305480[(unsigned __int8)(v13 - 14)];
  v16 = v37;
  if ( !(_BYTE)v37 )
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
  v32 = 0;
LABEL_8:
  v35 = v15;
  v19 = v17;
  v33 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v20) = sub_1D15020(v17, v15);
  v21 = 0;
  if ( !(_BYTE)v20 )
  {
    v20 = sub_1F593D0(v33, v19, v32, v35);
    v5 = v20;
    v21 = v31;
  }
  LOBYTE(v5) = v20;
  v36 = v21;
  v22 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v41,
          v5,
          v21,
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64,
          v39);
  v23 = *(__int64 **)(a1 + 8);
  v24 = *(unsigned __int16 *)(a2 + 24);
  *(_QWORD *)&v39 = v22;
  DWORD2(v39) = v25;
  v26 = sub_1D309E0(v23, v24, (__int64)&v41, v5, v36, 0, a3, a4, *(double *)a5.m128i_i64, v40);
  v27 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v40 = v26;
  DWORD2(v40) = v28;
  v29 = sub_1D332F0(
          v27,
          107,
          (__int64)&v41,
          v37,
          v38,
          0,
          a3,
          a4,
          a5,
          v39,
          *((unsigned __int64 *)&v39 + 1),
          __PAIR128__(*((unsigned __int64 *)&v40 + 1), v26));
  if ( v41 )
    sub_161E7C0((__int64)&v41, v41);
  return v29;
}
