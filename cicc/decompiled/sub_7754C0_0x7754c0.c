// Function: sub_7754C0
// Address: 0x7754c0
//
__int64 __fastcall sub_7754C0(__int64 a1, __int64 a2, __int64 a3, char **a4, _WORD *a5)
{
  char *v8; // rax
  char v9; // dl
  __int64 v10; // rcx
  __int64 v11; // r14
  unsigned int v12; // r15d
  unsigned __int8 v14; // dl
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r15
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rdx

  v8 = *a4;
  v9 = **a4;
  if ( v9 != 13 )
  {
    switch ( v9 )
    {
      case 2:
        goto LABEL_27;
      case 6:
        v17 = *((_QWORD *)v8 + 1);
        if ( v17 )
          goto LABEL_28;
        goto LABEL_7;
      case 7:
        goto LABEL_16;
      case 8:
        goto LABEL_33;
      case 37:
        v17 = *(_QWORD *)(*((_QWORD *)v8 + 1) + 40LL);
        if ( !v17 )
          goto LABEL_7;
        goto LABEL_28;
      default:
        goto LABEL_7;
    }
  }
  v10 = *((_QWORD *)v8 + 1);
  v14 = *(_BYTE *)(v10 + 24);
  if ( v14 == 4 )
  {
    *v8 = 8;
    v22 = *(_QWORD *)(v10 + 56);
    *((_QWORD *)v8 + 1) = v22;
    if ( (*(_BYTE *)(v22 - 8) & 1) != 0 )
      *((_DWORD *)v8 + 4) = 0;
LABEL_33:
    v23 = *((_QWORD *)v8 + 1);
    if ( (*(_BYTE *)(v23 + 144) & 4) == 0 )
    {
      v24 = sub_7A7D30(v23, 1);
      v11 = v24;
      if ( v24 )
        goto LABEL_24;
    }
    goto LABEL_7;
  }
  if ( v14 > 4u )
  {
    if ( v14 == 20 )
    {
      *v8 = 11;
      v10 = *(_QWORD *)(v10 + 56);
      *((_QWORD *)v8 + 1) = v10;
    }
    goto LABEL_5;
  }
  if ( v14 != 2 )
  {
    if ( v14 == 3 )
    {
      *v8 = 7;
      v15 = *(_QWORD *)(v10 + 56);
      *((_QWORD *)v8 + 1) = v15;
      if ( (*(_BYTE *)(v15 - 8) & 1) != 0 )
        *((_DWORD *)v8 + 4) = 0;
LABEL_16:
      v16 = *((_QWORD *)v8 + 1);
      v17 = *(_QWORD *)(v16 + 120);
      v18 = *(unsigned int *)(v16 + 152);
      if ( v17 )
      {
        if ( !(unsigned int)sub_8D2310(*(_QWORD *)(v16 + 120)) && !(unsigned int)sub_8D23B0(v17) || v18 )
          goto LABEL_19;
      }
      else if ( *(_DWORD *)(v16 + 152) )
      {
LABEL_23:
        v11 = v18;
LABEL_24:
        v12 = 1;
        goto LABEL_9;
      }
      goto LABEL_7;
    }
LABEL_5:
    if ( (*(_BYTE *)(v10 - 8) & 1) != 0 )
      *((_DWORD *)v8 + 4) = 0;
    goto LABEL_7;
  }
  *v8 = 2;
  v21 = *(_QWORD *)(v10 + 56);
  *((_QWORD *)v8 + 1) = v21;
  if ( (*(_BYTE *)(v21 - 8) & 1) != 0 )
    *((_DWORD *)v8 + 4) = 0;
LABEL_27:
  v17 = *(_QWORD *)(*((_QWORD *)v8 + 1) + 128LL);
  if ( !v17 )
    goto LABEL_7;
LABEL_28:
  if ( (unsigned int)sub_8D2310(v17) || (v18 = 0, (unsigned int)sub_8D23B0(v17)) )
  {
LABEL_7:
    v11 = 0;
    v12 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
    }
    goto LABEL_9;
  }
LABEL_19:
  v19 = *(_BYTE *)(v17 + 140);
  if ( v19 == 12 )
  {
    v20 = v17;
    do
    {
      v20 = *(_QWORD *)(v20 + 160);
      v19 = *(_BYTE *)(v20 + 140);
    }
    while ( v19 == 12 );
  }
  if ( !v19 )
    goto LABEL_23;
  if ( (unsigned int)sub_8D32E0(v17) )
    v17 = sub_8D46C0(v17);
  if ( *(char *)(v17 + 142) >= 0 && *(_BYTE *)(v17 + 140) == 12 )
  {
    v12 = 1;
    v11 = (unsigned int)sub_8D4AB0(v17, a2, v25);
  }
  else
  {
    v11 = *(unsigned int *)(v17 + 136);
    v12 = 1;
  }
LABEL_9:
  sub_620D80(a5, v11);
  return v12;
}
