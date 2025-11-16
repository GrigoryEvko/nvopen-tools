// Function: sub_169A460
// Address: 0x169a460
//
__int64 __fastcall sub_169A460(__int64 a1, char *a2, __int64 a3, unsigned int a4)
{
  char *v7; // rbx
  __int64 v8; // r14
  int v9; // eax
  char *v10; // rsi
  int v11; // r8d
  int v12; // eax
  char *v13; // r9
  int v14; // r8d
  char *v15; // rdx
  char v16; // di
  unsigned int v17; // r10d
  unsigned int v18; // ecx
  char *v19; // r11
  char v20; // di
  int v21; // r9d
  char *v22; // rax
  int v23; // edx
  int v24; // eax
  int v25; // eax
  __int64 v27; // rax
  char *i; // rcx
  char v29; // di
  char *v30; // rdx

  v7 = a2;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 2;
  sub_1698870(a1);
  *(_WORD *)(a1 + 16) = 0;
  v8 = sub_1698470(a1);
  v9 = sub_1698310(a1);
  v10 = &a2[a3];
  if ( v7 == &v7[a3] )
    goto LABEL_44;
  v11 = v9;
  while ( 1 )
  {
    v12 = *v7;
    if ( *v7 != 48 )
      break;
    if ( v10 == ++v7 )
      goto LABEL_44;
  }
  if ( v10 == v7 )
    goto LABEL_44;
  v13 = v10;
  if ( (_BYTE)v12 != 46 )
    goto LABEL_7;
  v30 = v7 + 1;
  if ( v10 == v7 + 1 )
    goto LABEL_44;
  while ( 1 )
  {
    v12 = *v30;
    if ( *v30 != 48 )
      break;
    if ( v10 == ++v30 )
      goto LABEL_44;
  }
  if ( v10 == v30 )
  {
LABEL_44:
    v17 = 0;
    return sub_1698EC0((__int16 **)a1, a4, v17);
  }
  v13 = v7;
  v7 = v30;
LABEL_7:
  v14 = v11 << 6;
  v15 = v7 + 1;
  v16 = 0;
  v17 = 0;
  v18 = v14;
  v19 = v7;
  if ( (_BYTE)v12 == 46 )
    goto LABEL_40;
LABEL_8:
  if ( (unsigned __int8)(v12 - 48) <= 9u )
  {
    v27 = (unsigned int)(char)(v12 - 48);
LABEL_24:
    if ( v18 )
    {
      v18 -= 4;
      *(_QWORD *)(v8 + 8LL * (v18 >> 6)) |= v27 << v18;
      goto LABEL_38;
    }
    if ( v16 )
      goto LABEL_38;
    if ( (unsigned int)v27 > 8 )
    {
      v16 = 1;
      v17 = 3;
LABEL_38:
      while ( v10 != v15 )
      {
LABEL_39:
        v12 = *v15++;
        v19 = v15 - 1;
        if ( (_BYTE)v12 != 46 )
          goto LABEL_8;
LABEL_40:
        v13 = v19;
      }
    }
    else
    {
      if ( (unsigned int)(v27 - 1) <= 6 )
      {
        v16 = 1;
        v17 = 1;
        goto LABEL_38;
      }
      if ( v10 == v15 )
      {
        v29 = *v10;
      }
      else
      {
        for ( i = v15; ; ++i )
        {
          v29 = *i;
          if ( ((*i - 46) & 0xFD) != 0 )
            break;
          if ( v10 == i + 1 )
          {
            v29 = i[1];
            break;
          }
        }
      }
      if ( (unsigned __int8)(v29 - 48) <= 9u || (unsigned __int8)((v29 & 0xDF) - 65) <= 5u )
      {
        v16 = 1;
        v18 = 0;
        v17 = (_DWORD)v27 == 0 ? 1 : 3;
        goto LABEL_38;
      }
      v16 = 1;
      v18 = 0;
      v17 = (_DWORD)v27 != 0 ? 2 : 0;
      if ( v10 != v15 )
        goto LABEL_39;
    }
    v19 = v10;
    goto LABEL_11;
  }
  if ( (unsigned __int8)(v12 - 97) <= 5u )
  {
    v27 = (unsigned int)(v12 - 87);
    goto LABEL_24;
  }
  if ( (unsigned __int8)(v12 - 65) <= 5u )
  {
    v27 = (unsigned int)(v12 - 55);
    goto LABEL_24;
  }
LABEL_11:
  if ( v19 != v7 )
  {
    v20 = v19[1];
    if ( v13 == v10 )
      LODWORD(v13) = (_DWORD)v19;
    v21 = (_DWORD)v13 - (_DWORD)v7 - (((unsigned int)((_DWORD)v13 - (_DWORD)v7) < 0x80000000) - 1);
    v22 = &v19[(((v20 - 43) & 0xFD) == 0) + 1];
    if ( v10 == v22 )
    {
      v23 = 0;
    }
    else
    {
      v23 = *v22 - 48;
      while ( v10 != ++v22 )
      {
        v23 = *v22 + 10 * v23 - 48;
        if ( v23 > 0x7FFF )
          goto LABEL_19;
      }
    }
    v24 = *(_DWORD *)(*(_QWORD *)a1 + 4LL) - v14 + 4 * v21 - 1;
    if ( (unsigned int)(v24 + 0x8000) <= 0xFFFF )
    {
      if ( v20 == 45 )
      {
        v25 = v24 - v23;
        if ( (unsigned int)(v25 + 0x8000) > 0xFFFF )
          LOWORD(v25) = 0x8000;
      }
      else
      {
        v25 = v23 + v24;
        if ( (unsigned int)(v25 + 0x8000) > 0xFFFF )
          LOWORD(v25) = 0x7FFF;
      }
    }
    else
    {
LABEL_19:
      LOWORD(v25) = 0x7FFF;
      if ( v20 == 45 )
        LOWORD(v25) = 0x8000;
    }
    *(_WORD *)(a1 + 16) = v25;
  }
  return sub_1698EC0((__int16 **)a1, a4, v17);
}
