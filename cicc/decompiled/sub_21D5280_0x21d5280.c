// Function: sub_21D5280
// Address: 0x21d5280
//
__int64 *__fastcall sub_21D5280(__m128i a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  unsigned int v7; // r11d
  __int64 v10; // rax
  __int64 v11; // r10
  int v12; // ecx
  __int64 v13; // r14
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rax
  __int64 *result; // rax
  unsigned int v19; // eax
  const void **v20; // rdx
  const void **v21; // r8
  __int64 v22; // rsi
  __int128 v23; // rax
  __int64 v24; // rcx
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // rbx
  __int128 v28; // rax
  __int128 v29; // rax
  __int128 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r9
  __int128 v33; // rax
  const void ***v34; // rsi
  __int64 v35; // r9
  __int128 v36; // [rsp-50h] [rbp-E0h]
  __int128 v37; // [rsp-30h] [rbp-C0h]
  unsigned int v38; // [rsp+0h] [rbp-90h]
  unsigned int v39; // [rsp+0h] [rbp-90h]
  __int128 v40; // [rsp+0h] [rbp-90h]
  __int64 v41; // [rsp+10h] [rbp-80h]
  unsigned int v42; // [rsp+10h] [rbp-80h]
  const void **v43; // [rsp+10h] [rbp-80h]
  __int128 v44; // [rsp+10h] [rbp-80h]
  unsigned int v45; // [rsp+20h] [rbp-70h]
  __int64 v46; // [rsp+30h] [rbp-60h]
  __int64 v47; // [rsp+38h] [rbp-58h]
  const void **v48; // [rsp+38h] [rbp-58h]
  const void **v49; // [rsp+38h] [rbp-58h]
  __int64 v50; // [rsp+38h] [rbp-58h]
  __int64 *v51; // [rsp+38h] [rbp-58h]
  char v52[8]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v53; // [rsp+48h] [rbp-48h]
  __int64 v54; // [rsp+50h] [rbp-40h] BYREF
  int v55; // [rsp+58h] [rbp-38h]

  v10 = *(_QWORD *)(a5 + 32);
  v11 = *(_QWORD *)(v10 + 40);
  v12 = *(unsigned __int16 *)(v11 + 24);
  if ( v12 == 32 || v12 == 10 )
    return (__int64 *)a5;
  v13 = *(_QWORD *)v10;
  v14 = *(_QWORD *)(v10 + 8);
  v45 = *(_DWORD *)(v10 + 48);
  v15 = *(_QWORD *)(*(_QWORD *)v10 + 40LL) + 16LL * *(unsigned int *)(v10 + 8);
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v52[0] = v16;
  v53 = v17;
  if ( v16 )
  {
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
        LOBYTE(v19) = 2;
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
        LOBYTE(v19) = 3;
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
        LOBYTE(v19) = 4;
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
        LOBYTE(v19) = 5;
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
        LOBYTE(v19) = 6;
        break;
      case 55:
        LOBYTE(v19) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v19) = 8;
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
        LOBYTE(v19) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v19) = 10;
        break;
    }
    v21 = 0;
  }
  else
  {
    v47 = v11;
    LOBYTE(v19) = sub_1F596B0((__int64)v52);
    v11 = v47;
    v7 = v19;
    v21 = v20;
  }
  v22 = *(_QWORD *)(a5 + 72);
  LOBYTE(v7) = v19;
  v54 = v22;
  if ( v22 )
  {
    v38 = v7;
    v41 = v11;
    v48 = v21;
    sub_1623A60((__int64)&v54, v22, 2);
    v7 = v38;
    v11 = v41;
    v21 = v48;
  }
  v46 = v11;
  v42 = v7;
  v49 = v21;
  v55 = *(_DWORD *)(a5 + 64);
  *(_QWORD *)&v23 = sub_1D38E70((__int64)a7, 0, (__int64)&v54, 0, a1, a2, a3);
  v24 = v42;
  v39 = v42;
  v43 = v49;
  v25 = sub_1D332F0(a7, 106, (__int64)&v54, v24, v49, 0, *(double *)a1.m128i_i64, a2, a3, v13, v14, v23);
  v50 = v26;
  v27 = v25;
  *(_QWORD *)&v28 = sub_1D38E70((__int64)a7, 1, (__int64)&v54, 0, a1, a2, a3);
  *(_QWORD *)&v29 = sub_1D332F0(a7, 106, (__int64)&v54, v39, v43, 0, *(double *)a1.m128i_i64, a2, a3, v13, v14, v28);
  v40 = v29;
  *(_QWORD *)&v30 = sub_1D38E70((__int64)a7, 0, (__int64)&v54, 0, a1, a2, a3);
  v44 = v30;
  *(_QWORD *)&v33 = sub_1D28D50(a7, 0x11u, 0, 0xFFFFFFFF00000000LL, v31, v32);
  v34 = (const void ***)(v27[5] + 16LL * (unsigned int)v50);
  *((_QWORD *)&v37 + 1) = v50;
  *(_QWORD *)&v37 = v27;
  *((_QWORD *)&v36 + 1) = v45;
  *(_QWORD *)&v36 = v46;
  result = sub_1D36A20(a7, 136, (__int64)&v54, *(unsigned __int8 *)v34, v34[1], v35, v36, v44, v37, v40, v33);
  if ( v54 )
  {
    v51 = result;
    sub_161E7C0((__int64)&v54, v54);
    return v51;
  }
  return result;
}
