// Function: sub_8DF240
// Address: 0x8df240
//
__int64 __fastcall sub_8DF240(__int64 a1, __int64 a2, int *a3, int a4, int a5, _DWORD *a6, __int64 *a7, __int64 *a8)
{
  int v10; // ebx
  int v11; // r12d
  unsigned int v12; // r13d
  __int64 v13; // rax
  bool v14; // zf
  int v15; // r12d
  int v16; // eax
  char v17; // dl
  char i; // al
  char v19; // al
  char v20; // bl
  char v21; // al
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r12
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // r8
  int v29; // eax
  bool v30; // al
  int v34; // [rsp+14h] [rbp-3Ch]
  int v36; // [rsp+1Ch] [rbp-34h]

  if ( a6 )
    *a6 = 0;
  v36 = 1;
  v34 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = 0;
        if ( (*(_BYTE *)(a2 + 140) & 0xFB) == 8 )
          v10 = sub_8D4C10(a2, dword_4F077C4 != 2);
        v11 = 0;
        if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
        {
          v11 = sub_8D4C10(a1, dword_4F077C4 != 2);
          if ( sub_8D3D40(a2) )
            goto LABEL_37;
        }
        else if ( sub_8D3D40(a2) )
        {
          goto LABEL_37;
        }
        v12 = sub_8D3D40(a1);
        if ( v12
          || sub_8D3410(a2) && (v13 = sub_8D4050(a2), sub_8D3D40(v13))
          || sub_8D3410(a1) && (v23 = sub_8D4050(a1), sub_8D3D40(v23)) )
        {
LABEL_37:
          v12 = 1;
          v20 = a6 != 0;
          goto LABEL_49;
        }
        if ( (v11 & ~v10) != 0 )
          goto LABEL_48;
        v14 = (v11 & 4) == 0;
        v15 = ~v11;
        if ( v14 )
          v10 &= ~4u;
        if ( (v10 & v15) != 0 )
        {
          if ( !v36 )
          {
            v34 = 1;
            v12 = 0;
            v20 = 0;
            goto LABEL_49;
          }
          v34 = v36;
        }
        v16 = 0;
        if ( (v10 & 1) != 0 )
          v16 = v36;
        v36 = v16;
        v17 = *(_BYTE *)(a2 + 140);
        if ( v17 != 12 )
          goto LABEL_24;
        do
        {
          a2 = *(_QWORD *)(a2 + 160);
          v17 = *(_BYTE *)(a2 + 140);
        }
        while ( v17 == 12 );
        for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(a1 + 140) )
        {
          a1 = *(_QWORD *)(a1 + 160);
LABEL_24:
          ;
        }
        if ( !a5 )
          break;
        if ( i != v17 )
          goto LABEL_37;
        if ( i == 6 )
        {
          if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
            goto LABEL_37;
          goto LABEL_73;
        }
        if ( i != 8 && i != 13 )
          goto LABEL_37;
        if ( v17 == 8 )
        {
          a2 = *(_QWORD *)(a2 + 160);
          v19 = *(_BYTE *)(a1 + 140);
          if ( dword_4D04964 )
          {
            if ( v19 == 8 )
              goto LABEL_35;
LABEL_33:
            if ( v19 == 6 && (*(_BYTE *)(a1 + 168) & 1) == 0 )
              goto LABEL_35;
LABEL_71:
            a1 = sub_8D4870(a1);
          }
          else
          {
            while ( *(_BYTE *)(a2 + 140) == 12 )
              a2 = *(_QWORD *)(a2 + 160);
            if ( v19 != 8 )
              goto LABEL_33;
            do
LABEL_77:
              a1 = *(_QWORD *)(a1 + 160);
            while ( *(_BYTE *)(a1 + 140) == 12 );
          }
        }
        else
        {
          if ( v17 != 6 )
            goto LABEL_32;
LABEL_73:
          if ( (*(_BYTE *)(a2 + 168) & 1) != 0 )
          {
LABEL_32:
            a2 = sub_8D4870(a2);
            v19 = *(_BYTE *)(a1 + 140);
            if ( v19 != 8 )
              goto LABEL_33;
            goto LABEL_75;
          }
          v19 = *(_BYTE *)(a1 + 140);
          a2 = *(_QWORD *)(a2 + 160);
          if ( v19 != 8 )
            goto LABEL_33;
LABEL_75:
          a1 = *(_QWORD *)(a1 + 160);
          if ( !dword_4D04964 && *(_BYTE *)(a1 + 140) == 12 )
            goto LABEL_77;
        }
      }
      if ( !(unsigned int)sub_8D2F30(a2, a1) )
        break;
      if ( *(_QWORD *)(a2 + 128) != *(_QWORD *)(a1 + 128) )
      {
        v12 = 0;
        v20 = 0;
LABEL_49:
        if ( a3 )
          goto LABEL_50;
        goto LABEL_51;
      }
      a2 = sub_8D46C0(a2);
      a1 = sub_8D46C0(a1);
    }
    v21 = *(_BYTE *)(a2 + 140);
    if ( v21 == 13 && *(_BYTE *)(a1 + 140) == 13 )
    {
      v25 = sub_8D4890(a1);
      v27 = sub_8D4890(a2);
      if ( v25 != v27 )
      {
        v12 = sub_8D97D0(v25, v27, 0, v26, v28);
        if ( !v12 )
        {
LABEL_48:
          v20 = 0;
          goto LABEL_49;
        }
      }
      a2 = sub_8D4870(a2);
      goto LABEL_71;
    }
    if ( !dword_4F077BC )
    {
      if ( (_DWORD)qword_4F077B4 )
        break;
      goto LABEL_41;
    }
    if ( (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1387Fu )
      break;
LABEL_41:
    if ( *(_BYTE *)(a1 + 140) != 8 || v21 != 8 )
      break;
    v12 = sub_8D1590(a1, a2);
    if ( !v12
      && (dword_4F077C4 != 2 || unk_4F07778 <= 202001 || (*(_WORD *)(a2 + 168) & 0x180) != 0 || *(_QWORD *)(a2 + 176)) )
    {
      goto LABEL_48;
    }
    a2 = *(_QWORD *)(a2 + 160);
LABEL_35:
    a1 = *(_QWORD *)(a1 + 160);
  }
  v20 = a6 != 0;
  if ( a4 )
  {
    v12 = 1;
    if ( !a3 )
    {
LABEL_58:
      if ( a7 )
        *a7 = a1;
      if ( a8 )
        *a8 = a2;
      goto LABEL_52;
    }
  }
  else
  {
    if ( a2 == a1 )
    {
      v12 = 1;
      v30 = 1;
    }
    else
    {
      v29 = sub_8DED30(a1, a2, 4194323, dword_4F077BC, v24);
      v14 = v29 == 0;
      v30 = v29 != 0;
      v12 = !v14;
    }
    v20 &= v30;
    if ( !a3 )
      goto LABEL_52;
  }
LABEL_50:
  *a3 = v34;
LABEL_51:
  if ( a4 )
    goto LABEL_58;
LABEL_52:
  if ( v20 )
    *a6 = 0;
  return v12;
}
