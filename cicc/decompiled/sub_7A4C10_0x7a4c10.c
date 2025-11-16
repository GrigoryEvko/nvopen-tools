// Function: sub_7A4C10
// Address: 0x7a4c10
//
__int64 __fastcall sub_7A4C10(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v7; // r14
  _QWORD *v8; // rdx
  __int64 i; // rax
  char v10; // cl
  __int64 j; // rax
  char v12; // dl
  __int64 v13; // rsi
  unsigned __int8 v14; // dl
  __int64 v15; // rdx
  FILE *v16; // r13
  unsigned int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  char v21; // dl
  __int64 v23; // rdx
  __int64 v24; // rdx
  char v25; // al
  _QWORD *v26; // rdx
  FILE *v27; // rsi
  unsigned int v28; // edi
  __int64 v29; // rax
  _QWORD *v31; // [rsp+18h] [rbp-58h] BYREF
  __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  __int64 v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h]

  v7 = *a4;
  v34 = 0;
  v33 = 0;
  v32 = sub_823970(0);
  v31 = sub_724DC0();
  v8 = v31;
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v10 = *(_BYTE *)(a1 + 132);
  if ( (v10 & 1) == 0 )
  {
    v17 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xAA1u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v8 = v31;
    }
    goto LABEL_29;
  }
  for ( j = *(_QWORD *)(***(_QWORD ***)(i + 168) + 8LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v12 = *(_BYTE *)v7;
  if ( *(_BYTE *)v7 == 48 )
  {
    v24 = *(_QWORD *)(v7 + 8);
    v25 = *(_BYTE *)(v24 + 8);
    if ( v25 == 1 )
    {
      *(_BYTE *)v7 = 2;
      *(_QWORD *)(v7 + 8) = *(_QWORD *)(v24 + 32);
      v10 = *(_BYTE *)(a1 + 132);
    }
    else if ( v25 == 2 )
    {
      *(_BYTE *)v7 = 59;
      *(_QWORD *)(v7 + 8) = *(_QWORD *)(v24 + 32);
      v10 = *(_BYTE *)(a1 + 132);
    }
    else
    {
      if ( v25 )
        goto LABEL_36;
      *(_BYTE *)v7 = 6;
      *(_QWORD *)(v7 + 8) = *(_QWORD *)(v24 + 32);
      v10 = *(_BYTE *)(a1 + 132);
    }
LABEL_41:
    if ( (v10 & 0x20) != 0 )
      goto LABEL_28;
    v26 = (_QWORD *)(a1 + 96);
    v27 = (FILE *)(a3 + 28);
LABEL_43:
    v28 = 3375;
LABEL_44:
    sub_6855B0(v28, v27, v26);
    sub_770D30(a1);
    goto LABEL_28;
  }
  if ( v12 != 13 )
    goto LABEL_16;
  v13 = *(_QWORD *)(v7 + 8);
  v14 = *(_BYTE *)(v13 + 24);
  if ( v14 == 4 )
  {
    *(_BYTE *)v7 = 8;
    v15 = *(_QWORD *)(v13 + 56);
    *(_QWORD *)(v7 + 8) = v15;
    goto LABEL_14;
  }
  if ( v14 <= 4u )
  {
    if ( v14 == 2 )
    {
      *(_BYTE *)v7 = 2;
      v15 = *(_QWORD *)(v13 + 56);
      *(_QWORD *)(v7 + 8) = v15;
      goto LABEL_14;
    }
    if ( v14 == 3 )
    {
      *(_BYTE *)v7 = 7;
      v15 = *(_QWORD *)(v13 + 56);
      *(_QWORD *)(v7 + 8) = v15;
LABEL_14:
      if ( (*(_BYTE *)(v15 - 8) & 1) == 0 )
      {
LABEL_57:
        v10 = *(_BYTE *)(a1 + 132);
        goto LABEL_41;
      }
      *(_DWORD *)(v7 + 16) = 0;
      v12 = *(_BYTE *)v7;
LABEL_16:
      if ( v12 == 11 )
        goto LABEL_17;
      goto LABEL_57;
    }
LABEL_53:
    if ( (*(_BYTE *)(v13 - 8) & 1) != 0 )
    {
      *(_DWORD *)(v7 + 16) = 0;
      v10 = *(_BYTE *)(a1 + 132);
    }
    goto LABEL_41;
  }
  if ( v14 != 20 )
    goto LABEL_53;
  *(_BYTE *)v7 = 11;
  v23 = *(_QWORD *)(v13 + 56);
  *(_QWORD *)(v7 + 8) = v23;
  if ( (*(_BYTE *)(v23 - 8) & 1) != 0 )
    *(_DWORD *)(v7 + 16) = 0;
LABEL_17:
  if ( **(_QWORD **)(j + 168) )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      goto LABEL_28;
    v26 = (_QWORD *)(a1 + 96);
    v27 = (FILE *)(a1 + 112);
    v28 = 3367;
    goto LABEL_44;
  }
  v16 = (FILE *)(a3 + 28);
  v17 = sub_7A4160(a1, *(_QWORD *)(j + 160), &v32, v16);
  if ( !v17 )
  {
LABEL_47:
    v8 = v31;
    goto LABEL_29;
  }
  v18 = v32;
  v19 = v32 + 24 * v34;
  if ( v19 != v32 )
  {
    while ( 1 )
    {
      while ( *(_BYTE *)v18 != 48 )
      {
        if ( *(_BYTE *)v18 != 2 )
          goto LABEL_27;
        v18 += 24;
        if ( v19 == v18 )
          goto LABEL_46;
      }
      v20 = *(_QWORD *)(v18 + 8);
      v21 = *(_BYTE *)(v20 + 8);
      if ( v21 != 1 )
        break;
      *(_BYTE *)v18 = 2;
      v18 += 24;
      *(_QWORD *)(v18 - 16) = *(_QWORD *)(v20 + 32);
      if ( v19 == v18 )
        goto LABEL_46;
    }
    if ( v21 == 2 )
    {
      *(_BYTE *)v18 = 59;
      *(_QWORD *)(v18 + 8) = *(_QWORD *)(v20 + 32);
LABEL_27:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      {
LABEL_28:
        v8 = v31;
        v17 = 0;
        goto LABEL_29;
      }
      v26 = (_QWORD *)(a1 + 96);
      v27 = v16;
      goto LABEL_43;
    }
    if ( !v21 )
    {
      *(_BYTE *)v18 = 6;
      *(_QWORD *)(v18 + 8) = *(_QWORD *)(v20 + 32);
      goto LABEL_27;
    }
LABEL_36:
    sub_721090();
  }
LABEL_46:
  v17 = sub_695090(v7, &v32, (__int64 *)&v16->_flags, (__int64)v31);
  if ( !v17 )
    goto LABEL_47;
  v17 = 1;
  *(_BYTE *)a5 = 2;
  v29 = sub_724E50((__int64 *)&v31, &v32);
  *(_DWORD *)(a5 + 16) = 0;
  v8 = v31;
  *(_QWORD *)(a5 + 8) = v29;
LABEL_29:
  if ( v8 )
    sub_724E30((__int64)&v31);
  sub_823A00(v32, 24 * v33);
  return v17;
}
