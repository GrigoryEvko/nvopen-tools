// Function: sub_732490
// Address: 0x732490
//
_BOOL8 __fastcall sub_732490(__int64 a1, unsigned int *a2)
{
  char v2; // al
  unsigned int v3; // eax
  _BOOL4 v4; // r8d
  __int64 v6; // rcx
  __int64 v7; // rdi
  char v8; // r9
  __int64 v9; // rax
  __int64 v10; // rax
  char v11; // r8
  __int64 v12; // rax

  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( (unsigned __int8)(v2 - 9) > 2u || (*(_BYTE *)(a1 + 177) & 0x20) != 0 )
    goto LABEL_6;
  if ( (*(_BYTE *)(a1 + 141) & 0x20) == 0 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
    if ( (*(_BYTE *)(v6 + 177) & 0x40) == 0 )
    {
      v4 = 1;
      v3 = dword_4D0425C;
      if ( !dword_4D0425C )
        goto LABEL_7;
      v3 = 0;
      if ( unk_4D04250 > 0x9E97u )
        goto LABEL_7;
      v7 = *(_QWORD *)(v6 + 8);
      v8 = *(_BYTE *)(v7 + 80);
      if ( v8 == 17 )
      {
        v12 = *(_QWORD *)(v7 + 88);
        if ( !v12 )
        {
          v3 = 0;
          goto LABEL_7;
        }
        while ( *(_BYTE *)(v12 + 80) != 10 || (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 194LL) & 4) == 0 )
        {
          v12 = *(_QWORD *)(v12 + 8);
          if ( !v12 )
            goto LABEL_20;
        }
      }
      else
      {
        v3 = 0;
        if ( v8 != 11 || (*(_BYTE *)(*(_QWORD *)(v7 + 88) + 194LL) & 4) == 0 )
          goto LABEL_7;
      }
    }
    if ( (*(_BYTE *)(v6 + 177) & 1) == 0 )
    {
      v9 = *(_QWORD *)(v6 + 24);
      if ( !v9 )
      {
LABEL_6:
        v3 = 0;
        v4 = 0;
LABEL_7:
        *a2 = v3;
        return v4;
      }
      v10 = *(_QWORD *)(v9 + 88);
      if ( (*(_BYTE *)(v10 + 194) & 8) != 0 )
      {
        v11 = *(_BYTE *)(v10 + 206);
        v3 = 0;
        v4 = (v11 & 0x10) != 0;
        goto LABEL_7;
      }
    }
LABEL_20:
    v3 = 0;
    v4 = 1;
    goto LABEL_7;
  }
  *a2 = 1;
  return 0;
}
