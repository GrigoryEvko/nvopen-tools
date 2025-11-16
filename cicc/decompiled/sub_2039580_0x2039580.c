// Function: sub_2039580
// Address: 0x2039580
//
__int64 *__fastcall sub_2039580(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r9
  char v16; // cl
  int v17; // r9d
  int v18; // eax
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  unsigned int v21; // eax
  const void **v22; // r8
  unsigned int v23; // ecx
  __int64 v24; // rsi
  __int64 *v25; // r15
  __int64 v26; // rsi
  __int64 *v27; // r12
  unsigned int v29; // edx
  const void **v30; // rdx
  unsigned int v31; // edx
  __int64 v32; // rax
  __int128 v33; // [rsp-10h] [rbp-D0h]
  __int64 v34; // [rsp+8h] [rbp-B8h]
  _QWORD *v35; // [rsp+10h] [rbp-B0h]
  unsigned int v36; // [rsp+1Ch] [rbp-A4h]
  __int64 v37; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v38; // [rsp+28h] [rbp-98h]
  char v39; // [rsp+30h] [rbp-90h]
  int v40; // [rsp+30h] [rbp-90h]
  unsigned int v41; // [rsp+30h] [rbp-90h]
  unsigned int v42; // [rsp+38h] [rbp-88h]
  __int64 v43; // [rsp+40h] [rbp-80h]
  __int64 v44; // [rsp+48h] [rbp-78h]
  unsigned int v45; // [rsp+50h] [rbp-70h] BYREF
  const void **v46; // [rsp+58h] [rbp-68h]
  _BYTE v47[8]; // [rsp+60h] [rbp-60h] BYREF
  const void **v48; // [rsp+68h] [rbp-58h]
  __int64 v49; // [rsp+70h] [rbp-50h] BYREF
  int v50; // [rsp+78h] [rbp-48h]
  const void **v51; // [rsp+80h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v49,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v45) = v50;
  v46 = v51;
  v6 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v7 = *a1;
  v37 = v6;
  v8 = *(_QWORD *)(a2 + 32);
  v38 = v9;
  v10 = *(_QWORD *)(v8 + 40);
  v11 = *(_QWORD *)(v8 + 48);
  v44 = v10;
  v43 = *(unsigned int *)(v8 + 48);
  v12 = *(_QWORD *)(v10 + 40) + 16 * v43;
  LOBYTE(v9) = *(_BYTE *)v12;
  v13 = *(_QWORD *)(v12 + 8);
  v14 = a1[1];
  v47[0] = v9;
  v15 = *(_QWORD *)(v14 + 48);
  v48 = (const void **)v13;
  sub_1F40D10((__int64)&v49, v7, v15, (unsigned __int8)v9, v13);
  v16 = v47[0];
  if ( (_BYTE)v49 == 7 )
  {
    v44 = sub_20363F0((__int64)a1, v10, v11);
    v32 = *(_QWORD *)(v44 + 40) + 16LL * v31;
    v11 = v31 | v11 & 0xFFFFFFFF00000000LL;
    v16 = *(_BYTE *)v32;
    v48 = *(const void ***)(v32 + 8);
    v47[0] = v16;
    v43 = v31;
  }
  if ( !(_BYTE)v45 )
  {
    v39 = v16;
    v18 = sub_1F58D30((__int64)&v45);
    v16 = v39;
    v17 = v18;
    if ( v39 )
      goto LABEL_5;
LABEL_7:
    v40 = v17;
    v19 = sub_1F596B0((__int64)v47);
    v17 = v40;
    v34 = v20;
    goto LABEL_8;
  }
  v17 = word_4305480[(unsigned __int8)(v45 - 14)];
  if ( !v16 )
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
      v19 = 2;
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
      v19 = 3;
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
      v19 = 4;
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
      v19 = 5;
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
      v19 = 6;
      break;
    case 55:
      v19 = 7;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      v19 = 8;
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
      v19 = 9;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      v19 = 10;
      break;
  }
  v34 = 0;
LABEL_8:
  v36 = v17;
  v41 = v19;
  v35 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v21) = sub_1D15020(v19, v17);
  v22 = 0;
  if ( !(_BYTE)v21 )
  {
    v21 = sub_1F593D0(v35, v41, v34, v36);
    v42 = v21;
    v22 = v30;
  }
  v23 = v42;
  LOBYTE(v23) = v21;
  if ( (_BYTE)v21 != v47[0] || !(_BYTE)v21 && v48 != v22 )
  {
    v11 = v43 | v11 & 0xFFFFFFFF00000000LL;
    v44 = (__int64)sub_2030300(a1, v44, v11, v23, v22, 0, a3, a4, a5);
    v43 = v29;
  }
  v24 = *(_QWORD *)(a2 + 72);
  v25 = (__int64 *)a1[1];
  v49 = v24;
  if ( v24 )
    sub_1623A60((__int64)&v49, v24, 2);
  v26 = *(unsigned __int16 *)(a2 + 24);
  v50 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v33 + 1) = v11 & 0xFFFFFFFF00000000LL | v43;
  *(_QWORD *)&v33 = v44;
  v27 = sub_1D332F0(v25, v26, (__int64)&v49, v45, v46, 0, *(double *)a3.m128i_i64, a4, a5, v37, v38, v33);
  if ( v49 )
    sub_161E7C0((__int64)&v49, v49);
  return v27;
}
