// Function: sub_8D67E0
// Address: 0x8d67e0
//
__int64 __fastcall sub_8D67E0(__int64 a1, __m128i *a2, __int64 a3, int a4, int *a5)
{
  char v8; // al
  char i; // cl
  _BOOL4 v10; // r8d
  __int64 result; // rax
  int v12; // edx
  char v13; // al
  char v14; // al
  unsigned __int64 v15; // r9
  unsigned __int64 v16; // rsi
  unsigned __int8 v17; // cl
  _QWORD *v18; // rax
  _QWORD *v19; // r13
  unsigned __int8 v20; // r15
  int v21; // eax
  __int64 j; // rax
  __int64 v23; // rsi
  __int64 v24; // rdi
  _QWORD *v25; // rax
  __int64 v26; // r13
  unsigned __int8 v27; // r15
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-68h]
  unsigned __int8 v30; // [rsp+1Bh] [rbp-55h] BYREF
  unsigned int v31; // [rsp+1Ch] [rbp-54h] BYREF
  int v32; // [rsp+20h] [rbp-50h] BYREF
  int v33; // [rsp+24h] [rbp-4Ch] BYREF
  _QWORD *v34; // [rsp+28h] [rbp-48h] BYREF
  _OWORD v35[4]; // [rsp+30h] [rbp-40h] BYREF

  v8 = *(_BYTE *)(a1 + 140);
  if ( v8 != 12 )
    goto LABEL_5;
  do
  {
    a1 = *(_QWORD *)(a1 + 160);
    v8 = *(_BYTE *)(a1 + 140);
  }
  while ( v8 == 12 );
  for ( i = *(_BYTE *)(a3 + 140); i == 12; i = *(_BYTE *)(a3 + 140) )
  {
    a3 = *(_QWORD *)(a3 + 160);
LABEL_5:
    ;
  }
  v10 = 0;
  if ( a2 )
    v10 = a2[10].m128i_i8[13] == 12;
  if ( (unsigned __int8)(v8 - 3) <= 2u )
  {
    if ( i != 2 )
    {
      if ( (i == v8 || (unsigned __int8)(v8 - 4) > 1u && (unsigned __int8)(i - 4) > 1u)
        && (unsigned __int8)(i - 3) <= 2u
        && *(_BYTE *)(a1 + 160) > *(_BYTE *)(a3 + 160) )
      {
        if ( !a2 || a2[10].m128i_i8[13] != 3 )
          goto LABEL_17;
        for ( j = a2[8].m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        sub_709EF0(a2 + 11, *(_BYTE *)(j + 160), v35, *(_BYTE *)(a3 + 160), &v31, &v32);
        if ( v31 )
          goto LABEL_65;
      }
      goto LABEL_22;
    }
    if ( !unk_4D04000 )
    {
      v13 = *(_BYTE *)(a3 + 161);
      if ( (v13 & 8) != 0 && (v13 & 0x14) == 0 )
      {
LABEL_22:
        if ( !a5 )
          return 0;
        goto LABEL_23;
      }
    }
LABEL_29:
    v12 = 2361;
    if ( a5 )
      goto LABEL_30;
    return 1;
  }
  if ( v8 != 2 || (*(_BYTE *)(a1 + 161) & 0x10) != 0 )
  {
    if ( i != 2 || (*(_BYTE *)(a3 + 162) & 4) == 0 )
      goto LABEL_22;
    if ( v8 == 6 )
    {
      if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
        goto LABEL_22;
    }
    else if ( (unsigned __int8)(v8 - 7) > 1u && v8 != 13 )
    {
      goto LABEL_22;
    }
    if ( (_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A0 <= 0x186A0u )
        goto LABEL_22;
    }
    else if ( dword_4F077BC && qword_4F077A8 <= 0x1869Fu )
    {
      goto LABEL_22;
    }
    goto LABEL_29;
  }
  if ( (unsigned __int8)(i - 3) <= 2u )
  {
    if ( !a2 || a2[10].m128i_i8[13] != 1 || i == 4 )
    {
LABEL_17:
      result = (unsigned int)(1 - v10);
      if ( !a5 )
        return result;
      v12 = 2361;
      if ( !(_DWORD)result )
        goto LABEL_23;
LABEL_30:
      result = 1;
      goto LABEL_24;
    }
    v29 = a3;
    v18 = sub_724DC0();
    v34 = v18;
    if ( *(_BYTE *)(v29 + 140) == 5 )
    {
      sub_724C70((__int64)v18, 4);
      v25 = v34;
      v34[16] = v29;
      v26 = v25[22];
      v27 = *(_BYTE *)(v29 + 160);
      v28 = sub_620E90((__int64)a2);
      sub_622780(a2 + 11, v28, v26, v27, &v31);
    }
    else
    {
      sub_724C70((__int64)v18, 3);
      v19 = v34;
      v34[16] = v29;
      v20 = *(_BYTE *)(v29 + 160);
      v21 = sub_620E90((__int64)a2);
      sub_622780(a2 + 11, v21, (__int64)(v19 + 22), v20, &v31);
    }
    if ( !v31 )
    {
      *(_QWORD *)&v35[0] = sub_724DC0();
      sub_724C70(*(__int64 *)&v35[0], 1);
      v23 = *(_QWORD *)&v35[0];
      v24 = (__int64)v34;
      *(_QWORD *)(*(_QWORD *)&v35[0] + 128LL) = a2[8].m128i_i64[0];
      sub_7103C0(v24, v23, &v33, &v30, &v32, 0);
      if ( v33 || (unsigned int)sub_621060((__int64)a2, *(__int64 *)&v35[0]) )
      {
        sub_724E30((__int64)v35);
        sub_724E30((__int64)&v34);
        if ( a5 )
        {
          result = 1;
          v12 = 2364;
          goto LABEL_24;
        }
        return 1;
      }
      sub_724E30((__int64)v35);
      sub_724E30((__int64)&v34);
      goto LABEL_22;
    }
    sub_724E30((__int64)&v34);
    goto LABEL_65;
  }
  if ( i != 2 )
    goto LABEL_22;
  if ( !unk_4D04000 )
  {
    v14 = *(_BYTE *)(a3 + 161);
    if ( (v14 & 8) != 0 && (!a4 || (v14 & 0x14) == 0) )
      goto LABEL_22;
  }
  if ( (*(_BYTE *)(a1 + 162) & 4) != 0 )
    goto LABEL_22;
  v15 = *(_QWORD *)(a1 + 128);
  v16 = *(_QWORD *)(a3 + 128);
  if ( v15 <= v16 && (*(_BYTE *)(a3 + 162) & 4) == 0 )
  {
    v17 = byte_4B6DF90[*(unsigned __int8 *)(a1 + 160)];
    if ( v15 != v16 )
    {
      if ( !v17 )
        goto LABEL_22;
LABEL_43:
      if ( byte_4B6DF90[*(unsigned __int8 *)(a3 + 160)] )
        goto LABEL_22;
      goto LABEL_44;
    }
    if ( v17 )
      goto LABEL_43;
    if ( !byte_4B6DF90[*(unsigned __int8 *)(a3 + 160)] )
      goto LABEL_22;
  }
LABEL_44:
  if ( a2 && a2[10].m128i_i8[13] == 1 )
  {
    if ( (*(_BYTE *)(a3 + 162) & 4) != 0 )
    {
      if ( dword_4F077BC || !(unsigned int)sub_6210B0((__int64)a2, 0) || !(unsigned int)sub_6210B0((__int64)a2, 1) )
        goto LABEL_22;
    }
    else if ( sub_621140((__int64)a2, (__int64)a2, *(_BYTE *)(a3 + 160)) )
    {
      goto LABEL_22;
    }
LABEL_65:
    v12 = 2362;
    if ( a5 )
      goto LABEL_30;
    return 1;
  }
  result = (unsigned int)(1 - v10);
  if ( a5 )
  {
    if ( (_DWORD)result )
    {
      v12 = 2361;
      goto LABEL_30;
    }
LABEL_23:
    result = 0;
    v12 = 0;
LABEL_24:
    *a5 = v12;
  }
  return result;
}
