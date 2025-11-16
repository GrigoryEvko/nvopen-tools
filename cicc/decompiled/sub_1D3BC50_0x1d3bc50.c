// Function: sub_1D3BC50
// Address: 0x1d3bc50
//
__int64 *__fastcall sub_1D3BC50(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v10; // rbx
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rcx
  unsigned __int8 v14; // r8
  __int64 v15; // rdx
  char v16; // di
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  unsigned int v23; // eax
  unsigned __int64 v24; // rax
  __int128 v25; // rax
  __int64 *result; // rax
  char v27; // al
  __int64 v28; // rdx
  __int64 v29; // rcx
  char v30; // al
  char v31; // al
  __int64 v32; // rcx
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // r8
  unsigned int v36; // eax
  char v37; // al
  unsigned __int8 v38; // [rsp+7h] [rbp-89h]
  __int64 v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+8h] [rbp-88h]
  unsigned int v41; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+10h] [rbp-80h]
  unsigned __int8 v43; // [rsp+10h] [rbp-80h]
  unsigned int v44; // [rsp+10h] [rbp-80h]
  __int64 *v47; // [rsp+20h] [rbp-70h]
  __int64 v48; // [rsp+30h] [rbp-60h] BYREF
  __int64 v49; // [rsp+38h] [rbp-58h]
  char v50[8]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v51; // [rsp+48h] [rbp-48h]
  unsigned __int64 v52; // [rsp+50h] [rbp-40h] BYREF
  __int64 v53; // [rsp+58h] [rbp-38h]

  v10 = 16LL * (unsigned int)a3;
  v11 = *(_QWORD *)(a2 + 40);
  v48 = a5;
  v49 = a6;
  v12 = *(_BYTE *)(v11 + v10);
  v13 = *(_QWORD *)(v11 + v10 + 8);
  LOBYTE(v52) = v12;
  v53 = v13;
  if ( !v12 )
  {
    v39 = v11;
    v42 = v13;
    v30 = sub_1F58D20(&v52);
    v29 = v42;
    v11 = v39;
    if ( v30 )
    {
      v27 = sub_1F596B0(&v52);
      v14 = v48;
      v29 = v28;
      if ( (_BYTE)v48 != v27 )
      {
LABEL_24:
        v11 = *(_QWORD *)(a2 + 40);
        goto LABEL_4;
      }
      if ( (_BYTE)v48 )
        return (__int64 *)a2;
    }
    else
    {
      v14 = v48;
      if ( (_BYTE)v48 )
        goto LABEL_4;
    }
    if ( v49 != v29 )
    {
      v14 = 0;
      goto LABEL_24;
    }
    return (__int64 *)a2;
  }
  if ( (unsigned __int8)(v12 - 14) <= 0x5Fu )
  {
    v14 = v48;
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
        if ( (_BYTE)v48 != 3 )
          goto LABEL_4;
        return (__int64 *)a2;
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
        if ( (_BYTE)v48 != 4 )
          goto LABEL_4;
        return (__int64 *)a2;
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
        if ( (_BYTE)v48 != 5 )
          goto LABEL_4;
        return (__int64 *)a2;
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
        if ( (_BYTE)v48 != 6 )
          goto LABEL_4;
        return (__int64 *)a2;
      case 55:
        if ( (_BYTE)v48 != 7 )
          goto LABEL_4;
        return (__int64 *)a2;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        if ( (_BYTE)v48 != 8 )
          goto LABEL_4;
        return (__int64 *)a2;
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
        if ( (_BYTE)v48 != 9 )
          goto LABEL_4;
        return (__int64 *)a2;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        if ( (_BYTE)v48 != 10 )
          goto LABEL_4;
        return (__int64 *)a2;
      default:
        if ( (_BYTE)v48 != 2 )
          goto LABEL_4;
        return (__int64 *)a2;
    }
  }
  v14 = v48;
  if ( v12 == (_BYTE)v48 )
    return (__int64 *)a2;
LABEL_4:
  v15 = v10 + v11;
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v50[0] = v16;
  v51 = v17;
  if ( v16 )
  {
    if ( (unsigned __int8)(v16 - 14) <= 0x5Fu )
    {
      switch ( v16 )
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
          v16 = 3;
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
          v16 = 4;
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
          v16 = 5;
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
          v16 = 6;
          break;
        case 55:
          v16 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v16 = 8;
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
          v16 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v16 = 10;
          break;
        default:
          v16 = 2;
          break;
      }
    }
LABEL_6:
    v18 = sub_1D13440(v16);
    v22 = v18;
    v41 = v18;
    if ( (_BYTE)v20 )
      goto LABEL_7;
LABEL_28:
    v23 = sub_1F58D40(&v48, a2, v22, v19, v20, v21);
    LODWORD(v53) = v41;
    if ( v41 <= 0x40 )
      goto LABEL_8;
LABEL_29:
    v44 = v23;
    sub_16A4EF0((__int64)&v52, 0, 0);
    v23 = v44;
    goto LABEL_9;
  }
  v38 = v14;
  v40 = v17;
  v31 = sub_1F58D20(v50);
  v34 = v40;
  v35 = v38;
  if ( v31 )
  {
    v37 = sub_1F596B0(v50);
    v35 = (unsigned __int8)v48;
    LOBYTE(v52) = v37;
    v16 = v37;
    v53 = v34;
    if ( v37 )
      goto LABEL_6;
  }
  else
  {
    LOBYTE(v52) = 0;
    v53 = v40;
  }
  v43 = v35;
  v36 = sub_1F58D40(&v52, a2, v34, v32, v35, v33);
  v20 = v43;
  v22 = v36;
  v41 = v36;
  if ( !(_BYTE)v20 )
    goto LABEL_28;
LABEL_7:
  v23 = sub_1D13440(v20);
  LODWORD(v53) = v41;
  if ( v41 > 0x40 )
    goto LABEL_29;
LABEL_8:
  v52 = 0;
LABEL_9:
  if ( v23 )
  {
    if ( v23 > 0x40 )
    {
      sub_16A5260(&v52, 0, v23);
    }
    else
    {
      v24 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23);
      if ( (unsigned int)v53 > 0x40 )
        *(_QWORD *)v52 |= v24;
      else
        v52 |= v24;
    }
  }
  *(_QWORD *)&v25 = sub_1D38970(
                      (__int64)a1,
                      (__int64)&v52,
                      a4,
                      *(unsigned __int8 *)(v10 + *(_QWORD *)(a2 + 40)),
                      *(const void ***)(v10 + *(_QWORD *)(a2 + 40) + 8),
                      0,
                      a7,
                      a8,
                      a9,
                      0);
  result = sub_1D332F0(
             a1,
             118,
             a4,
             *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + v10),
             *(const void ***)(*(_QWORD *)(a2 + 40) + v10 + 8),
             0,
             *(double *)a7.m128i_i64,
             a8,
             a9,
             a2,
             a3,
             v25);
  if ( (unsigned int)v53 > 0x40 )
  {
    if ( v52 )
    {
      v47 = result;
      j_j___libc_free_0_0(v52);
      return v47;
    }
  }
  return result;
}
