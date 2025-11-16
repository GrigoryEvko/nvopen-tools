// Function: sub_38ACF80
// Address: 0x38acf80
//
__int64 __fastcall sub_38ACF80(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  __int16 v6; // r14
  __int64 v7; // r13
  int v9; // eax
  unsigned int v10; // eax
  int v11; // eax
  const char *v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v15; // rdi
  const char *v16; // rax
  unsigned int v17; // eax
  _QWORD *v18; // rax
  _QWORD *v19; // r12
  unsigned __int64 v20; // [rsp+8h] [rbp-88h]
  __int16 v21; // [rsp+14h] [rbp-7Ch]
  unsigned __int64 v22; // [rsp+18h] [rbp-78h]
  char v23; // [rsp+2Bh] [rbp-65h] BYREF
  int v24; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v26; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v27[2]; // [rsp+40h] [rbp-50h] BYREF
  char v28; // [rsp+50h] [rbp-40h]
  char v29; // [rsp+51h] [rbp-3Fh]

  v6 = 0;
  v7 = a1 + 8;
  v9 = *(_DWORD *)(a1 + 64);
  v24 = 0;
  v23 = 1;
  if ( v9 == 66 )
  {
    v6 = 1;
    v11 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v11;
    v10 = v11 - 227;
    if ( v10 > 0x17 )
    {
LABEL_5:
      v29 = 1;
      v12 = "expected binary operation in atomicrmw";
LABEL_6:
      v13 = *(_QWORD *)(a1 + 56);
      v27[0] = v12;
      v28 = 3;
      return (unsigned __int8)sub_38814C0(v7, v13, (__int64)v27);
    }
  }
  else
  {
    v10 = v9 - 227;
  }
  switch ( v10 )
  {
    case 0u:
      v21 = 0;
      break;
    case 1u:
      v21 = 4;
      break;
    case 2u:
      v21 = 7;
      break;
    case 3u:
      v21 = 8;
      break;
    case 4u:
      v21 = 9;
      break;
    case 5u:
      v21 = 10;
      break;
    case 6u:
      v21 = 1;
      break;
    case 8u:
      v21 = 2;
      break;
    case 0x15u:
      v21 = 3;
      break;
    case 0x16u:
      v21 = 5;
      break;
    case 0x17u:
      v21 = 6;
      break;
    default:
      goto LABEL_5;
  }
  *(_DWORD *)(a1 + 64) = sub_3887100(v7);
  v22 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v25, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after atomicrmw address") )
    return 1;
  v20 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v26, a3, a4, a5, a6)
    || (unsigned __int8)sub_388CFF0(a1, 1u, &v23, &v24) )
  {
    return 1;
  }
  if ( v24 == 1 )
  {
    v29 = 1;
    v12 = "atomicrmw cannot be unordered";
    goto LABEL_6;
  }
  if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) == 15 )
  {
    v15 = *v26;
    if ( *v26 != *(_QWORD *)(*(_QWORD *)v25 + 24LL) )
    {
      v29 = 1;
      v16 = "atomicrmw value and pointer type do not match";
LABEL_16:
      v27[0] = v16;
      v28 = 3;
      return (unsigned __int8)sub_38814C0(v7, v20, (__int64)v27);
    }
    if ( *(_BYTE *)(v15 + 8) != 11 )
    {
      v29 = 1;
      v16 = "atomicrmw operand must be an integer";
      goto LABEL_16;
    }
    v17 = sub_1643030(v15);
    if ( v17 <= 7 || (v17 & (v17 - 1)) != 0 )
    {
      v29 = 1;
      v16 = "atomicrmw operand must be power-of-two byte-sized integer";
      goto LABEL_16;
    }
    v18 = sub_1648A60(64, 2u);
    v19 = v18;
    if ( v18 )
      sub_15F9C10((__int64)v18, v21, v25, v26, v24, v23, 0);
    *((_WORD *)v19 + 9) = *((_WORD *)v19 + 9) & 0xFFFE | v6;
    *a2 = v19;
    return 0;
  }
  else
  {
    v29 = 1;
    v28 = 3;
    v27[0] = "atomicrmw operand must be a pointer";
    return (unsigned __int8)sub_38814C0(v7, v22, (__int64)v27);
  }
}
