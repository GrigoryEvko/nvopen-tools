// Function: sub_7CFB70
// Address: 0x7cfb70
//
__int64 __fastcall sub_7CFB70(_QWORD *a1, char a2)
{
  int v2; // r15d
  int v3; // ebx
  __int64 v4; // r12
  char v6; // al
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r9
  char v11; // cl
  int v12; // r14d
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  __int64 v15; // rcx
  unsigned __int8 v16; // si
  char v17; // si
  __int64 v18; // r9
  __int64 v19; // rdi
  char v20; // si
  unsigned __int8 v21; // dl
  __int64 v22; // rcx
  unsigned __int8 v23; // si
  char v24; // dl

  v2 = a2 & 2;
  v3 = dword_4F077C4;
  if ( dword_4F077C4 != 2 )
    v3 = (v2 == 0) + 1;
  if ( (*((_BYTE *)a1 + 17) & 0x20) != 0 )
    return 0;
  v4 = a1[3];
  if ( !v4 )
  {
    v8 = (int)dword_4F04C5C;
    if ( dword_4F04C5C == -1 )
      BUG();
    v9 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
    v10 = *(_QWORD *)(v9 + 184);
    if ( unk_4D03F98 && v10 )
    {
      if ( *(_QWORD *)(*a1 + 64LL) )
      {
        v11 = *(_BYTE *)(v10 + 28);
        if ( !v11 || v11 == 3 )
        {
          sub_824D70(*(_QWORD *)(v9 + 184));
          v8 = (int)dword_4F04C5C;
          if ( dword_4F04C5C == -1 )
            BUG();
        }
      }
      v9 = qword_4F04C68[0] + 776 * v8;
    }
    v12 = a2 & 0x40;
    v13 = *(_QWORD *)(*a1 + 24LL);
    if ( v13 )
    {
      while ( 1 )
      {
        v14 = *(_BYTE *)(v13 + 80);
        v15 = v13;
        v16 = v14;
        if ( v14 == 16 )
        {
          v15 = **(_QWORD **)(v13 + 88);
          v16 = *(_BYTE *)(v15 + 80);
        }
        if ( v16 == 24 )
          v15 = *(_QWORD *)(v15 + 88);
        if ( *(_DWORD *)(v13 + 40) == *(_DWORD *)v9 )
        {
          if ( !v2
            || (v17 = *(_BYTE *)(v15 + 80), (unsigned __int8)(v17 - 4) <= 2u)
            || v17 == 3 && *(_BYTE *)(v15 + 104) )
          {
            if ( v3 == dword_4F04BA0[v14] && (v12 || v14 != 16) )
              break;
          }
        }
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          goto LABEL_37;
      }
LABEL_33:
      a1[3] = v13;
      v4 = v13;
      goto LABEL_5;
    }
LABEL_37:
    if ( (*(_BYTE *)(v9 + 4) & 0xFB) != 0 )
      goto LABEL_60;
    v18 = *(_QWORD *)(v9 + 24);
    v19 = v9 + 32;
    if ( !v18 )
      v18 = v19;
    v13 = sub_883800(v18, *a1);
    if ( !v13 )
    {
LABEL_60:
      a1[3] = 0;
      return v4;
    }
    do
    {
      v21 = *(_BYTE *)(v13 + 80);
      v22 = v13;
      v23 = v21;
      if ( v21 == 16 )
      {
        v22 = **(_QWORD **)(v13 + 88);
        v23 = *(_BYTE *)(v22 + 80);
      }
      if ( v23 == 24 )
        v22 = *(_QWORD *)(v22 + 88);
      if ( !v2 || (v20 = *(_BYTE *)(v22 + 80), (unsigned __int8)(v20 - 4) <= 2u) || v20 == 3 && *(_BYTE *)(v22 + 104) )
      {
        if ( v3 == dword_4F04BA0[v21] && (v12 || v21 != 16) )
        {
          v24 = *(_BYTE *)(v22 + 80);
          if ( (unsigned __int8)(v24 - 4) > 2u && (v24 != 3 || !*(_BYTE *)(v22 + 104)) )
            goto LABEL_33;
          if ( v4 )
          {
            if ( *(_BYTE *)(v4 + 80) == 24 )
              v4 = v13;
          }
          else
          {
            v4 = v13;
          }
        }
      }
      v13 = *(_QWORD *)(v13 + 32);
    }
    while ( v13 );
    a1[3] = v4;
    if ( v4 )
      goto LABEL_5;
    return 0;
  }
LABEL_5:
  v6 = *(_BYTE *)(v4 + 80);
  if ( v6 == 16 )
  {
    v4 = **(_QWORD **)(v4 + 88);
    v6 = *(_BYTE *)(v4 + 80);
  }
  if ( v6 == 24 )
    return *(_QWORD *)(v4 + 88);
  return v4;
}
