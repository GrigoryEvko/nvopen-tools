// Function: sub_2039A20
// Address: 0x2039a20
//
__int64 *__fastcall sub_2039A20(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v6; // r13d
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  unsigned int v12; // r12d
  unsigned int v13; // eax
  unsigned __int64 v14; // r8
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // r10
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rsi
  __int64 *v29; // r12
  unsigned __int64 v31; // rdx
  __int128 v32; // [rsp-10h] [rbp-90h]
  __int64 v33; // [rsp+0h] [rbp-80h]
  unsigned __int64 v34; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  _QWORD *v37; // [rsp+10h] [rbp-70h]
  unsigned int v38; // [rsp+18h] [rbp-68h]
  unsigned int v39; // [rsp+18h] [rbp-68h]
  __int64 *v40; // [rsp+18h] [rbp-68h]
  unsigned int v41; // [rsp+20h] [rbp-60h] BYREF
  const void **v42; // [rsp+28h] [rbp-58h]
  __int64 v43; // [rsp+30h] [rbp-50h] BYREF
  __int64 v44; // [rsp+38h] [rbp-48h]
  const void **v45; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v43,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v41) = v44;
  v42 = v45;
  if ( (_BYTE)v44 )
    v6 = word_4305480[(unsigned __int8)(v44 - 14)];
  else
    v6 = sub_1F58D30((__int64)&v41);
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  v8 = *(_BYTE *)(v7 + 88);
  v9 = *(_QWORD *)(v7 + 96);
  LOBYTE(v43) = v8;
  v44 = v9;
  if ( v8 )
  {
    switch ( v8 )
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
        v10 = 2;
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
        v10 = 3;
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
        v10 = 4;
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
        v10 = 5;
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
        v10 = 6;
        break;
      case 55:
        v10 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v10 = 8;
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
        v10 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v10 = 10;
        break;
    }
    v33 = 0;
  }
  else
  {
    v10 = sub_1F596B0((__int64)&v43);
    v33 = v11;
  }
  v12 = v10;
  v37 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v13) = sub_1D15020(v10, v6);
  v14 = 0;
  if ( !(_BYTE)v13 )
  {
    v13 = sub_1F593D0(v37, v12, v33, v6);
    v38 = v13;
    v14 = v31;
  }
  v15 = v38;
  v34 = v14;
  LOBYTE(v15) = v13;
  v39 = v15;
  v16 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v17 = v39;
  v19 = v18;
  v20 = v16;
  v40 = (__int64 *)a1[1];
  v23 = sub_1D2EF30(v40, v17, v34, v21, v34, v22);
  v25 = v40;
  v26 = v23;
  v27 = v24;
  v43 = *(_QWORD *)(a2 + 72);
  if ( v43 )
  {
    v36 = v24;
    v35 = v23;
    sub_1623A60((__int64)&v43, v43, 2);
    v26 = v35;
    v27 = v36;
    v25 = v40;
  }
  *((_QWORD *)&v32 + 1) = v27;
  v28 = *(unsigned __int16 *)(a2 + 24);
  *(_QWORD *)&v32 = v26;
  LODWORD(v44) = *(_DWORD *)(a2 + 64);
  v29 = sub_1D332F0(v25, v28, (__int64)&v43, v41, v42, 0, a3, a4, a5, v20, v19, v32);
  if ( v43 )
    sub_161E7C0((__int64)&v43, v43);
  return v29;
}
