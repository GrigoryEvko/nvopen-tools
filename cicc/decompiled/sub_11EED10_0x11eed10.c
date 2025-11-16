// Function: sub_11EED10
// Address: 0x11eed10
//
void __fastcall sub_11EED10(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // r13
  char v11; // al
  unsigned int v12; // ecx
  __int64 v13; // r8
  __int64 v14; // r9
  char v15; // al
  __int64 v16; // rdx
  __int64 v19; // [rsp-50h] [rbp-50h]
  __int64 v21; // [rsp-48h] [rbp-48h]
  unsigned int v22; // [rsp-40h] [rbp-40h]
  unsigned int v23; // [rsp-2Ch] [rbp-2Ch] BYREF

  if ( *(_BYTE *)a2 != 85 )
    return;
  if ( !*(_QWORD *)(a2 + 16) )
    return;
  if ( a3 != sub_B43CB0(a2) )
    return;
  v8 = sub_B43CA0(a2);
  v9 = *(_QWORD *)(a2 - 32);
  v10 = (__int64 *)v8;
  if ( !v9 )
    return;
  if ( *(_BYTE *)v9 )
    return;
  if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(a2 + 80) )
    return;
  if ( !sub_981210(**(_QWORD **)(a1 + 24), v9, &v23) )
    return;
  if ( !sub_11C99B0(v10, *(__int64 **)(a1 + 24), v23) )
    return;
  v11 = sub_A73ED0((_QWORD *)(a2 + 72), 41);
  v12 = a4;
  v13 = a5;
  v14 = a6;
  if ( !v11 )
  {
    v15 = sub_B49560(a2, 41);
    v12 = a4;
    v13 = a5;
    v14 = a6;
    if ( !v15 )
      return;
  }
  v19 = v14;
  v21 = v13;
  v22 = v12;
  if ( !sub_B49E00(a2) )
    return;
  if ( !(_BYTE)v22 )
  {
    if ( v23 != 134 )
    {
      if ( v23 != 86 )
      {
        if ( v23 == 129 )
          goto LABEL_17;
        return;
      }
LABEL_22:
      sub_11EECC0(v19, a2, v16, v22, v21, v19);
      return;
    }
LABEL_23:
    sub_11EECC0(v21, a2, v16, v22, v21, v19);
    return;
  }
  switch ( v23 )
  {
    case 0x87u:
      goto LABEL_23;
    case 0x57u:
      goto LABEL_22;
    case 0x82u:
LABEL_17:
      sub_11EECC0(a7, a2, v16, v22, v21, v19);
      break;
  }
}
