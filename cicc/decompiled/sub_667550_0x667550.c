// Function: sub_667550
// Address: 0x667550
//
void __fastcall sub_667550(__int64 *a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r12
  char v3; // dl
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // cl
  __int64 v7; // r13
  __int64 *i; // r12
  char v9; // al
  _QWORD *j; // rax
  char v11; // al
  __int64 v12; // rax
  char *v13; // rdx

  if ( (*((_BYTE *)a1 + 10) & 0x40) == 0 )
    return;
  v1 = a1;
  v2 = *a1;
  if ( !*a1
    || (v3 = *(_BYTE *)(v2 + 80), (unsigned __int8)(v3 - 4) <= 1u)
    || v3 == 3 && ((a1 = *(__int64 **)(v2 + 88), (unsigned int)sub_8D3A70(a1)) || (v3 = *(_BYTE *)(v2 + 80), v3 == 3)) )
  {
    v5 = v1[36];
    if ( v5 && (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) <= 2u && !*(_QWORD *)(v5 + 8) )
    {
      v9 = *(_BYTE *)(*(_QWORD *)(v5 + 168) + 113LL);
      if ( v9 )
      {
        if ( v9 != 1 )
          sub_721090(a1);
      }
      else if ( dword_4F077C4 == 2 )
      {
        sub_684AA0(7, 2501, (char *)v1 + 260);
      }
      return;
    }
LABEL_12:
    sub_6851C0(2501, (char *)v1 + 260);
    return;
  }
  if ( (*(_BYTE *)(v2 + 81) & 0x20) != 0 )
    return;
  v4 = v1[36];
  if ( v4 )
  {
    while ( 1 )
    {
      v6 = *(_BYTE *)(v4 + 140);
      if ( v6 != 12 )
        break;
      v4 = *(_QWORD *)(v4 + 160);
    }
    if ( !v6 )
      return;
  }
  if ( ((v3 - 7) & 0xFD) != 0 )
  {
    if ( v3 == 8 && (*(_BYTE *)(*(_QWORD *)(v2 + 88) + 144LL) & 0x10) != 0 && !dword_4D04964 )
      return;
    goto LABEL_12;
  }
  v7 = *(_QWORD *)(v2 + 88);
  if ( (v1[28] & 1) != 0 )
    sub_6851C0(1379, (char *)v1 + 260);
  if ( *((_BYTE *)v1 + 268) > 2u )
    sub_6851C0(1378, (char *)v1 + 260);
  for ( i = *(__int64 **)(v7 + 104); i; i = (__int64 *)*i )
  {
    if ( *((_BYTE *)i + 8) == 47 )
      sub_684B30(2505, i + 7);
  }
  if ( dword_4F077C4 != 2 && dword_4F04C58 != -1 && (unsigned __int8)(*((_BYTE *)v1 + 268) - 1) > 1u )
    sub_6851C0(2545, (char *)v1 + 260);
  *(_BYTE *)(v7 + 176) |= 8u;
  if ( (*(_BYTE *)(v7 + 170) & 2) != 0 )
  {
    for ( j = *(_QWORD **)(v7 + 128); j; j = (_QWORD *)*j )
      *(_BYTE *)(j[2] + 176LL) |= 8u;
  }
  v11 = *(_BYTE *)(v7 + 156);
  if ( (v11 & 1) != 0 )
  {
    v13 = "__constant__";
    if ( (v11 & 4) == 0 )
    {
      v13 = "__managed__";
      if ( (*(_BYTE *)(v7 + 157) & 1) == 0 )
      {
        v13 = "__shared__";
        if ( (v11 & 2) == 0 )
          v13 = "__device__";
      }
    }
    sub_6851A0(3570, (char *)v1 + 260, v13);
  }
  else if ( (*(_BYTE *)(v7 + 89) & 1) != 0 )
  {
    if ( unk_4F04C50 )
    {
      v12 = *(_QWORD *)(unk_4F04C50 + 32LL);
      if ( v12 )
      {
        if ( (*(_BYTE *)(v12 + 198) & 0x10) != 0 )
          sub_6851C0(3571, (char *)v1 + 260);
      }
    }
  }
}
