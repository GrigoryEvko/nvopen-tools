// Function: sub_6EC0E0
// Address: 0x6ec0e0
//
__int64 __fastcall sub_6EC0E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v4; // r15
  __int64 v7; // rcx
  int v8; // ebx
  __int64 v9; // r8
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rbx
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // r10
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-40h]
  __int64 v31; // [rsp+0h] [rbp-40h]
  __int64 v32; // [rsp+0h] [rbp-40h]
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+0h] [rbp-40h]

  v4 = a1;
  v8 = sub_8D2E30(*a1);
  if ( !v8 )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)v4 + 24) != 1 )
        return 0;
      if ( (unsigned __int8)(*((_BYTE *)v4 + 56) - 8) > 1u )
        break;
      while ( 1 )
      {
        v17 = (_QWORD *)v4[9];
        if ( *v4 == *v17 )
          break;
        if ( !(unsigned int)sub_8D97D0(*v4, *v17, 32, v7, v9) )
          goto LABEL_23;
        v4 = (_QWORD *)v4[9];
        if ( *((_BYTE *)v4 + 24) != 1 )
          return 0;
        if ( (unsigned __int8)(*((_BYTE *)v4 + 56) - 8) > 1u )
          goto LABEL_23;
      }
      v4 = (_QWORD *)v4[9];
    }
LABEL_23:
    if ( *((_BYTE *)v4 + 24) != 1 || *((_BYTE *)v4 + 56) != 14 )
      return 0;
    v18 = *v4;
    if ( v8 )
    {
      v18 = sub_8D46C0(*v4);
      if ( *(_BYTE *)(v18 + 140) != 12 )
      {
        v19 = *(_QWORD *)v4[9];
        goto LABEL_38;
      }
    }
    else if ( *(_BYTE *)(v18 + 140) != 12 )
    {
      v19 = *(_QWORD *)v4[9];
      goto LABEL_29;
    }
    do
      v18 = *(_QWORD *)(v18 + 160);
    while ( *(_BYTE *)(v18 + 140) == 12 );
    v19 = *(_QWORD *)v4[9];
    if ( !v8 )
    {
LABEL_29:
      if ( *(_BYTE *)(v19 + 140) != 12 )
      {
        v34 = v18;
        v28 = sub_73CA70(v19, a3);
        v21 = v34;
        v22 = v28;
        goto LABEL_32;
      }
      goto LABEL_30;
    }
LABEL_38:
    v32 = v18;
    v25 = sub_8D46C0(v19);
    v18 = v32;
    v19 = v25;
    if ( *(_BYTE *)(v25 + 140) != 12 )
    {
      v26 = sub_73CA70(v25, a3);
      v21 = v32;
      v22 = v26;
      goto LABEL_40;
    }
    do
LABEL_30:
      v19 = *(_QWORD *)(v19 + 160);
    while ( *(_BYTE *)(v19 + 140) == 12 );
    v30 = v18;
    v20 = sub_73CA70(v19, a3);
    v21 = v30;
    v22 = v20;
    if ( !v8 )
    {
LABEL_32:
      v31 = v22;
      if ( (*(_BYTE *)(sub_8D5CE0(v19, v21) + 96) & 2) != 0 )
        return 0;
      v15 = sub_73DBF0(15, v31, 0);
      sub_730580(v4, v15);
      *(_BYTE *)(v15 + 27) |= 2u;
      if ( a2 == v19 || (unsigned int)sub_8D97D0(v19, a2, 0, v23, v24) )
      {
        *a4 = v15;
        return v15;
      }
      v29 = sub_6EC0E0(v4[9], a2, a3, a4);
      if ( v29 )
      {
        *(_QWORD *)(v29 + 72) = v15;
        return v15;
      }
      return 0;
    }
LABEL_40:
    v33 = v21;
    v27 = sub_72D2E0(v22, 0);
    v21 = v33;
    v22 = v27;
    goto LABEL_32;
  }
  if ( *((_BYTE *)a1 + 24) == 1 )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)v4 + 56) != 5 || !(unsigned int)sub_8D2E30(*v4) || !(unsigned int)sub_8D2E30(*(_QWORD *)v4[9]) )
        goto LABEL_23;
      v10 = sub_8D46C0(*v4);
      v11 = sub_8D46C0(*(_QWORD *)v4[9]);
      v14 = v11;
      if ( *(_BYTE *)(v10 + 140) != 12 )
        goto LABEL_10;
      do
        v10 = *(_QWORD *)(v10 + 160);
      while ( *(_BYTE *)(v10 + 140) == 12 );
      if ( *(_BYTE *)(v11 + 140) == 12 )
        break;
LABEL_11:
      if ( v10 != v14 && !(unsigned int)sub_8D97D0(v10, v14, 0, v12, v13) )
        goto LABEL_23;
      v4 = (_QWORD *)v4[9];
      if ( *((_BYTE *)v4 + 24) != 1 )
        return 0;
    }
    while ( 1 )
    {
      v14 = *(_QWORD *)(v14 + 160);
LABEL_10:
      if ( *(_BYTE *)(v14 + 140) != 12 )
        goto LABEL_11;
    }
  }
  return 0;
}
