// Function: sub_774220
// Address: 0x774220
//
__int64 __fastcall sub_774220(__int64 a1, __int64 a2, __int64 a3, char **a4, _WORD *a5)
{
  char *v7; // rdx
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r15d
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rax
  char v18; // al
  __int64 v19; // rsi
  __int64 v20; // rax

  v7 = *a4;
  v8 = **a4;
  if ( v8 != 13 )
  {
    switch ( v8 )
    {
      case 2:
        goto LABEL_40;
      case 6:
        v10 = *((_QWORD *)v7 + 1);
        goto LABEL_8;
      case 7:
        goto LABEL_19;
      case 8:
        goto LABEL_5;
      case 37:
        v10 = *(_QWORD *)(*((_QWORD *)v7 + 1) + 40LL);
        goto LABEL_8;
      default:
        goto LABEL_10;
    }
  }
  v13 = *((_QWORD *)v7 + 1);
  v14 = *(_BYTE *)(v13 + 24);
  if ( v14 == 4 )
  {
    *v7 = 8;
    v9 = *(_QWORD *)(v13 + 56);
    *((_QWORD *)v7 + 1) = v9;
    if ( (*(_BYTE *)(v9 - 8) & 1) != 0 )
    {
      *((_DWORD *)v7 + 4) = 0;
LABEL_5:
      v9 = *((_QWORD *)v7 + 1);
    }
    if ( (*(_BYTE *)(v9 + 144) & 4) != 0 )
    {
      v11 = 1;
      sub_620D80(a5, 0);
      return v11;
    }
    goto LABEL_7;
  }
  if ( v14 > 4u )
  {
    if ( v14 == 20 )
    {
      *v7 = 11;
      v15 = *(_QWORD *)(v13 + 56);
      *((_QWORD *)v7 + 1) = v15;
      if ( (*(_BYTE *)(v15 - 8) & 1) != 0 )
        *((_DWORD *)v7 + 4) = 0;
      goto LABEL_10;
    }
LABEL_42:
    if ( (*(_BYTE *)(v13 - 8) & 1) != 0 )
      *((_DWORD *)v7 + 4) = 0;
    v10 = *(_QWORD *)v13;
    goto LABEL_8;
  }
  if ( v14 != 2 )
  {
    if ( v14 == 3 )
    {
      *v7 = 7;
      v9 = *(_QWORD *)(v13 + 56);
      *((_QWORD *)v7 + 1) = v9;
      if ( (*(_BYTE *)(v9 - 8) & 1) != 0 )
      {
        *((_DWORD *)v7 + 4) = 0;
LABEL_19:
        v9 = *((_QWORD *)v7 + 1);
      }
LABEL_7:
      v10 = *(_QWORD *)(v9 + 120);
      goto LABEL_8;
    }
    goto LABEL_42;
  }
  *v7 = 2;
  v20 = *(_QWORD *)(v13 + 56);
  *((_QWORD *)v7 + 1) = v20;
  if ( (*(_BYTE *)(v20 - 8) & 1) != 0 )
  {
    *((_DWORD *)v7 + 4) = 0;
LABEL_40:
    v20 = *((_QWORD *)v7 + 1);
  }
  v10 = *(_QWORD *)(v20 + 128);
LABEL_8:
  if ( v10 && !(unsigned int)sub_8D2310(v10) && !(unsigned int)sub_8D23B0(v10) )
  {
    v16 = *(_BYTE *)(v10 + 140);
    if ( v16 == 12 )
    {
      v17 = v10;
      do
      {
        v17 = *(_QWORD *)(v17 + 160);
        v16 = *(_BYTE *)(v17 + 140);
      }
      while ( v16 == 12 );
    }
    v11 = 1;
    if ( v16 )
    {
      if ( (unsigned int)sub_8D32E0(v10) )
        v10 = sub_8D46C0(v10);
      v18 = *(_BYTE *)(v10 + 140);
      if ( v18 == 12 )
      {
        v19 = sub_8D4A00(v10);
      }
      else if ( dword_4F077C0 && (v18 == 1 || v18 == 7) )
      {
        v19 = 1;
      }
      else
      {
        v19 = *(_QWORD *)(v10 + 128);
      }
      v11 = 1;
      sub_620D80(a5, v19);
    }
    return v11;
  }
LABEL_10:
  v11 = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
  }
  return v11;
}
