// Function: sub_1D360F0
// Address: 0x1d360f0
//
_QWORD *__fastcall sub_1D360F0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9)
{
  unsigned __int8 v9; // r10
  char v11; // r13
  const void **v14; // r11
  int v15; // r15d
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const void **v21; // r11
  unsigned __int8 v22; // r10
  _QWORD *v23; // r13
  char v25; // al
  const void **v26; // rdx
  __int128 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rcx
  unsigned __int8 *v31; // rsi
  __int64 *v32; // rdx
  __int64 v33; // rax
  unsigned __int8 v34; // [rsp+0h] [rbp-100h]
  const void **v35; // [rsp+0h] [rbp-100h]
  const void **v36; // [rsp+8h] [rbp-F8h]
  __int64 v37; // [rsp+8h] [rbp-F8h]
  unsigned __int8 v38; // [rsp+8h] [rbp-F8h]
  __int64 v39; // [rsp+10h] [rbp-F0h]
  int v40; // [rsp+10h] [rbp-F0h]
  __int64 v41; // [rsp+20h] [rbp-E0h] BYREF
  const void **v42; // [rsp+28h] [rbp-D8h]
  __int64 *v43; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v44; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v45[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v46[176]; // [rsp+50h] [rbp-B0h] BYREF

  v9 = a4;
  v11 = a6;
  v41 = a4;
  v42 = a5;
  if ( (_BYTE)a4 )
  {
    if ( (unsigned __int8)(a4 - 14) > 0x5Fu )
    {
LABEL_3:
      v14 = v42;
      goto LABEL_4;
    }
    switch ( (char)a4 )
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
        v9 = 3;
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
        v9 = 4;
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
        v9 = 5;
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
        v9 = 6;
        break;
      case 55:
        v9 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v9 = 8;
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
        v9 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v9 = 10;
        break;
      default:
        v9 = 2;
        break;
    }
    v14 = 0;
  }
  else
  {
    v25 = sub_1F58D20(&v41);
    v9 = 0;
    if ( !v25 )
      goto LABEL_3;
    v9 = sub_1F596B0(&v41);
    v14 = v26;
  }
LABEL_4:
  v45[0] = (unsigned __int64)v46;
  v15 = v11 == 0 ? 11 : 33;
  v34 = v9;
  v36 = v14;
  v45[1] = 0x2000000000LL;
  v39 = sub_1D29190((__int64)a1, v9, (__int64)v14, a4, (__int64)a5, a6);
  sub_16BD430((__int64)v45, v15);
  sub_16BD4C0((__int64)v45, v39);
  sub_16BD4C0((__int64)v45, a2);
  v43 = 0;
  v16 = sub_1D17920((__int64)a1, (__int64)v45, a3, (__int64 *)&v43);
  v21 = v36;
  v22 = v34;
  v23 = v16;
  if ( !v16 )
  {
    v23 = (_QWORD *)a1[26];
    v40 = *(_DWORD *)(a3 + 8);
    if ( v23 )
    {
      a1[26] = *v23;
    }
    else
    {
      v35 = v36;
      v38 = v22;
      v33 = sub_145CBF0(a1 + 27, 112, 8);
      v22 = v38;
      v21 = v35;
      v23 = (_QWORD *)v33;
    }
    *((_QWORD *)&v27 + 1) = v21;
    *(_QWORD *)&v27 = v22;
    v28 = sub_1D274F0(v27, v17, v18, v19, v20);
    v29 = *(_QWORD *)a3;
    v30 = v28;
    v44 = (unsigned __int8 *)v29;
    if ( v29 )
    {
      v37 = v28;
      sub_1623A60((__int64)&v44, v29, 2);
      v30 = v37;
    }
    *v23 = 0;
    v31 = v44;
    v23[7] = 0x100000000LL;
    v23[1] = 0;
    v23[2] = 0;
    *((_WORD *)v23 + 12) = v15;
    *((_DWORD *)v23 + 7) = -1;
    v23[4] = 0;
    v23[5] = v30;
    v23[6] = 0;
    *((_DWORD *)v23 + 16) = v40;
    v23[9] = v31;
    if ( v31 )
      sub_1623210((__int64)&v44, v31, (__int64)(v23 + 9));
    v23[11] = a2;
    v32 = v43;
    *((_WORD *)v23 + 40) &= 0xF000u;
    *((_WORD *)v23 + 13) = 0;
    sub_16BDA20(a1 + 40, v23, v32);
    sub_1D172A0((__int64)a1, (__int64)v23);
  }
  if ( (_BYTE)v41 )
  {
    if ( (unsigned __int8)(v41 - 14) > 0x5Fu )
      goto LABEL_7;
  }
  else if ( !(unsigned __int8)sub_1F58D20(&v41) )
  {
    goto LABEL_7;
  }
  v23 = sub_1D35F20(a1, (unsigned int)v41, v42, a3, (__int64)v23, 0, a7, a8, a9);
LABEL_7:
  if ( (_BYTE *)v45[0] != v46 )
    _libc_free(v45[0]);
  return v23;
}
