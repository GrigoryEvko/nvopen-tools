// Function: sub_14DA760
// Address: 0x14da760
//
__int64 __fastcall sub_14DA760(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // eax
  int v15; // ebx
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r13
  __int64 v20; // r14
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned __int8 *v24; // r14
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rdx
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // r13
  char v32; // dl
  char v33; // al
  __int64 v34; // r13
  char v35; // al
  float v36; // xmm0_4
  char v37; // al
  char v38; // al
  __int64 v39; // r13
  _QWORD *v40; // r14
  int *v41; // rax
  int *v42; // rbx
  double v43; // xmm0_8
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  double v47; // xmm1_8
  __int64 v48; // r13
  char v49; // al
  double x; // [rsp+0h] [rbp-90h]
  double xa; // [rsp+0h] [rbp-90h]
  _QWORD v52[2]; // [rsp+8h] [rbp-88h] BYREF
  unsigned int v53; // [rsp+1Ch] [rbp-74h] BYREF
  _BYTE v54[8]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v55[3]; // [rsp+28h] [rbp-68h] BYREF
  __int64 v56; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v57[9]; // [rsp+48h] [rbp-48h] BYREF

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (a1 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  v52[0] = a1;
  if ( (a1 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 21) )
    {
      v11 = *(_QWORD *)(v2 - 24);
      if ( *(_BYTE *)(v11 + 16) )
        goto LABEL_4;
      v56 = *(_QWORD *)(v11 + 112);
      if ( !(unsigned __int8)sub_1560260(&v56, 0xFFFFFFFFLL, 21) )
        goto LABEL_4;
    }
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 5) )
      goto LABEL_4;
    v9 = *(_QWORD *)(v2 - 24);
    if ( *(_BYTE *)(v9 + 16) )
      goto LABEL_6;
  }
  else
  {
    if ( !(unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 21) )
    {
      v8 = *(_QWORD *)(v2 - 72);
      if ( *(_BYTE *)(v8 + 16) )
        goto LABEL_4;
      v56 = *(_QWORD *)(v8 + 112);
      if ( !(unsigned __int8)sub_1560260(&v56, 0xFFFFFFFFLL, 21) )
        goto LABEL_4;
    }
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 5) )
      goto LABEL_4;
    v9 = *(_QWORD *)(v2 - 72);
    if ( *(_BYTE *)(v9 + 16) )
      goto LABEL_6;
  }
  v56 = *(_QWORD *)(v9 + 112);
  if ( !(unsigned __int8)sub_1560260(&v56, 0xFFFFFFFFLL, 5) )
    goto LABEL_6;
LABEL_4:
  v5 = v52[0] & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (v52[0] & 0xFFFFFFFFFFFFFFF8LL) + 56;
  if ( (v52[0] & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v6, 0xFFFFFFFFLL, 52) )
    {
LABEL_6:
      LODWORD(v5) = 0;
      return (unsigned int)v5;
    }
    v10 = *(_QWORD *)(v5 - 24);
    if ( !*(_BYTE *)(v10 + 16) )
      goto LABEL_24;
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v6, 0xFFFFFFFFLL, 52) )
      goto LABEL_6;
    v10 = *(_QWORD *)(v5 - 72);
    if ( !*(_BYTE *)(v10 + 16) )
    {
LABEL_24:
      v56 = *(_QWORD *)(v10 + 112);
      if ( (unsigned __int8)sub_1560260(&v56, 0xFFFFFFFFLL, 52) )
        goto LABEL_6;
    }
  }
  v12 = (v52[0] & 0xFFFFFFFFFFFFFFF8LL) - 24;
  if ( (v52[0] & 4) == 0 )
    v12 = (v52[0] & 0xFFFFFFFFFFFFFFF8LL) - 72;
  v13 = *(_QWORD *)v12;
  LOBYTE(v5) = a2 == 0 || *(_BYTE *)(*(_QWORD *)v12 + 16LL) != 0;
  if ( (_BYTE)v5 )
    goto LABEL_6;
  LOBYTE(v14) = sub_149CB50(*a2, v13, &v53);
  v15 = v14;
  if ( !(_BYTE)v14 )
    goto LABEL_6;
  if ( (unsigned int)sub_14DA610(v52) != 1 )
    goto LABEL_34;
  v16 = v52[0] & 0xFFFFFFFFFFFFFFF8LL;
  v17 = *(_DWORD *)((v52[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
  v18 = 4 * v17;
  v19 = *(_QWORD *)((v52[0] & 0xFFFFFFFFFFFFFFF8LL) - 24 * v17);
  if ( *(_BYTE *)(v19 + 16) != 14 )
    goto LABEL_34;
  v20 = v19 + 24;
  if ( v53 > 0x11A )
  {
    switch ( v53 )
    {
      case 0x158u:
      case 0x159u:
      case 0x15Du:
LABEL_69:
        if ( *(_QWORD *)(v19 + 32) == sub_16982C0(v52, v13, v16, v18) )
          v39 = *(_QWORD *)(v19 + 40) + 8LL;
        else
          v39 = v19 + 32;
        LOBYTE(v5) = (*(_BYTE *)(v39 + 18) & 7) != 0;
        return (unsigned int)v5;
      case 0x15Au:
      case 0x15Bu:
      case 0x15Cu:
LABEL_61:
        v37 = *(_BYTE *)(*(_QWORD *)v19 + 8LL);
        if ( v37 == 3 )
        {
          sub_14D1B20((__int64)v54, -710.0);
          if ( !(unsigned int)sub_14A9E40(v20, (__int64)v54) )
            goto LABEL_54;
          v5 = (unsigned __int64)&v56;
          sub_14D1B20((__int64)&v56, 710.0);
        }
        else
        {
          if ( v37 != 2 )
            goto LABEL_34;
          sub_14D1B70((__int64)v54, v13, -89.0);
          if ( !(unsigned int)sub_14A9E40(v20, (__int64)v54) )
          {
LABEL_54:
            sub_127D120(v55);
            return (unsigned int)v5;
          }
          v36 = 89.0;
          v5 = (unsigned __int64)&v56;
LABEL_59:
          sub_14D1B70((__int64)&v56, (__int64)v54, v36);
        }
LABEL_60:
        LOBYTE(v5) = (unsigned int)sub_14A9E40(v20, (__int64)&v56) != 2;
        sub_127D120(v57);
        goto LABEL_54;
      case 0x161u:
      case 0x162u:
      case 0x163u:
        if ( *(_QWORD *)(v19 + 32) != sub_16982C0(v52, v13, v16, v18) )
        {
          if ( (*(_BYTE *)(v19 + 50) & 5) != 1 )
          {
LABEL_51:
            v34 = v19 + 32;
            goto LABEL_52;
          }
          goto LABEL_79;
        }
        v48 = *(_QWORD *)(v19 + 40);
        if ( (*(_BYTE *)(v48 + 26) & 5) == 1 )
          goto LABEL_79;
        goto LABEL_86;
      case 0x189u:
      case 0x18Au:
      case 0x18Eu:
        v40 = *(_QWORD **)v19;
        if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v19 + 8LL) - 1) > 2u )
          goto LABEL_34;
        x = sub_14D1620((_QWORD *)v19, v13, v16, v18);
        feclearexcept(61);
        v41 = __errno_location();
        *v41 = 0;
        v42 = v41;
        v43 = tan(x);
        if ( fetestexcept(29) )
        {
          feclearexcept(61);
          *v42 = 0;
          v44 = 0;
        }
        else
        {
          v44 = sub_14D17B0(v40, v13, v43);
        }
        goto LABEL_76;
      default:
        goto LABEL_34;
    }
  }
  if ( v53 > 0x77 )
  {
    switch ( v53 )
    {
      case 0x78u:
      case 0x79u:
      case 0x7Du:
      case 0x7Eu:
      case 0x7Fu:
      case 0x83u:
        sub_169E660(v54, *(_QWORD *)(v19 + 32), "-1", 2);
        if ( !(unsigned int)sub_14A9E40(v19 + 24, (__int64)v54) )
          goto LABEL_54;
        v5 = (unsigned __int64)&v56;
        sub_169E660(&v56, *(_QWORD *)(v19 + 32), "1", 1);
        goto LABEL_60;
      case 0xA5u:
      case 0xA6u:
      case 0xAAu:
        goto LABEL_69;
      case 0xA7u:
      case 0xA8u:
      case 0xA9u:
        goto LABEL_61;
      case 0xACu:
      case 0xB3u:
      case 0xB4u:
        v38 = *(_BYTE *)(*(_QWORD *)v19 + 8LL);
        if ( v38 != 3 )
        {
          if ( v38 != 2 )
            goto LABEL_34;
          sub_14D1B70((__int64)v54, v13, -103.0);
          if ( (unsigned int)sub_14A9E40(v20, (__int64)v54) )
          {
            v36 = 88.0;
            v5 = (unsigned __int64)&v56;
            goto LABEL_59;
          }
          goto LABEL_54;
        }
        sub_14D1B20((__int64)v54, -745.0);
        if ( !(unsigned int)sub_14A9E40(v20, (__int64)v54) )
          goto LABEL_54;
        v5 = (unsigned __int64)&v56;
        sub_14D1B20((__int64)&v56, 709.0);
        goto LABEL_60;
      case 0xB0u:
      case 0xB1u:
      case 0xB2u:
        v35 = *(_BYTE *)(*(_QWORD *)v19 + 8LL);
        if ( v35 != 3 )
        {
          if ( v35 != 2 )
            goto LABEL_34;
          sub_14D1B70((__int64)v54, v13, -149.0);
          if ( (unsigned int)sub_14A9E40(v20, (__int64)v54) )
          {
            v36 = 127.0;
            v5 = (unsigned __int64)&v56;
            goto LABEL_59;
          }
          goto LABEL_54;
        }
        sub_14D1B20((__int64)v54, -1074.0);
        if ( !(unsigned int)sub_14A9E40(v20, (__int64)v54) )
          goto LABEL_54;
        v5 = (unsigned __int64)&v56;
        sub_14D1B20((__int64)&v56, 1023.0);
        goto LABEL_60;
      case 0x10Cu:
      case 0x10Du:
      case 0x10Eu:
      case 0x10Fu:
      case 0x113u:
      case 0x114u:
      case 0x115u:
      case 0x119u:
      case 0x11Au:
        if ( *(_QWORD *)(v19 + 32) != sub_16982C0(v52, v13, v16, v18) )
        {
          v33 = *(_BYTE *)(v19 + 50) & 7;
          if ( v33 == 1 )
            goto LABEL_79;
          if ( v33 == 3 )
            goto LABEL_6;
          goto LABEL_51;
        }
        v48 = *(_QWORD *)(v19 + 40);
        v49 = *(_BYTE *)(v48 + 26) & 7;
        if ( v49 == 1 )
          goto LABEL_79;
        if ( v49 == 3 )
          goto LABEL_6;
        break;
      default:
        goto LABEL_34;
    }
LABEL_86:
    v34 = v48 + 8;
LABEL_52:
    LODWORD(v5) = ((*(_BYTE *)(v34 + 18) >> 3) ^ 1) & 1;
    return (unsigned int)v5;
  }
LABEL_34:
  if ( (unsigned int)sub_14DA610(v52) != 2 )
    goto LABEL_6;
  v21 = v52[0] & 0xFFFFFFFFFFFFFFF8LL;
  v22 = *(_DWORD *)((v52[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
  v23 = 4 * v22;
  v24 = *(unsigned __int8 **)((v52[0] & 0xFFFFFFFFFFFFFFF8LL) - 24 * v22);
  v25 = v24[16];
  if ( (_BYTE)v25 != 14 )
    v24 = 0;
  v26 = 1 - v22;
  v27 = *(_QWORD *)(v21 + 24 * (1 - v22));
  if ( *(_BYTE *)(v27 + 16) != 14 || !v24 )
    goto LABEL_6;
  if ( v53 > 0xD9 )
  {
    if ( v53 - 313 > 2 )
      return (unsigned int)v5;
    v5 = *(_QWORD *)v24;
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v24 + 8LL) - 1) > 2u || v5 != *(_QWORD *)v27 )
      goto LABEL_6;
    xa = sub_14D1620(v24, v23, v26, v25);
    v47 = sub_14D1620((_QWORD *)v27, v23, v45, v46);
    v44 = sub_14D1A80(j_pow, (_QWORD *)v5, xa, v47);
LABEL_76:
    LOBYTE(v5) = v44 != 0;
  }
  else if ( v53 > 0xD6 )
  {
    v28 = sub_16982C0(v21, v23, v26, v25);
    v29 = (__int64)(v24 + 32);
    if ( *((_QWORD *)v24 + 4) == v28 )
      v29 = *((_QWORD *)v24 + 5) + 8LL;
    v30 = *(_BYTE *)(v29 + 18) & 7;
    if ( (_BYTE)v30 == 1
      || (v28 == *(_QWORD *)(v27 + 32) ? (v31 = *(_QWORD *)(v27 + 40) + 8LL) : (v31 = v27 + 32),
          v32 = *(_BYTE *)(v31 + 18) & 7,
          v32 == 1) )
    {
LABEL_79:
      LODWORD(v5) = v15;
    }
    else
    {
      LOBYTE(v5) = (_BYTE)v30 == 0;
      LOBYTE(v30) = v32 == 3;
      LODWORD(v5) = (v30 | v5) ^ 1;
    }
  }
  return (unsigned int)v5;
}
