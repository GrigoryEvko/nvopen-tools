// Function: sub_34CC460
// Address: 0x34cc460
//
unsigned __int64 __fastcall sub_34CC460(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned __int64 v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  char (__fastcall *v10)(__int64, __int64); // rax
  int v11; // eax
  unsigned __int16 v12; // dx
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  int v18; // eax
  char v19; // al
  int v20; // ecx
  __int64 v21; // rax
  char v22; // dl
  char v23; // al

  v2 = sub_2E313E0(a1);
  v3 = *(_QWORD *)(a1 + 56);
  v4 = v2;
  v5 = v2;
  if ( v2 == v3 )
    return v4;
  do
  {
    v6 = (_QWORD *)(*(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL);
    v7 = v6;
    if ( !v6 )
      BUG();
    v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
    v8 = *v6;
    if ( (v8 & 4) == 0 && (*((_BYTE *)v7 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        v5 = v9;
        if ( (*(_BYTE *)(v9 + 44) & 4) == 0 )
          break;
        v8 = *(_QWORD *)v9;
      }
    }
  }
  while ( v3 != v5 && (unsigned __int16)(*(_WORD *)(v5 + 68) - 14) <= 4u );
  v10 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 1328LL);
  if ( v10 != sub_2FDE950 )
  {
    v19 = ((__int64 (__fastcall *)(_DWORD *, __int64, _QWORD *))v10)(a2, v4, v7);
    goto LABEL_37;
  }
  v11 = *(_DWORD *)(v4 + 44);
  if ( (v11 & 4) == 0 && (v11 & 8) != 0 )
  {
    if ( sub_2E88A90(v4, 32, 1) )
      goto LABEL_34;
LABEL_13:
    v12 = *(_WORD *)(v5 + 68);
    goto LABEL_14;
  }
  if ( (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) & 0x20LL) == 0 )
    goto LABEL_13;
LABEL_34:
  v18 = *(_DWORD *)(v4 + 44);
  if ( (v18 & 4) == 0 && (v18 & 8) != 0 )
    v19 = sub_2E88A90(v4, 128, 1);
  else
    v19 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 7;
LABEL_37:
  if ( !v19 )
    goto LABEL_13;
  v12 = *(_WORD *)(v5 + 68);
  if ( v12 != a2[17] )
  {
LABEL_14:
    if ( v12 == 20 )
      goto LABEL_25;
LABEL_15:
    if ( v12 == 10 )
    {
      while ( 1 )
      {
LABEL_25:
        v16 = *(_QWORD *)(v5 + 32);
        if ( *(_BYTE *)v16
          || (*(_BYTE *)(v16 + 3) & 0x10) == 0
          || v12 != 10
          && (*(_BYTE *)(v16 + 40)
           || (unsigned int)(*(_DWORD *)(v16 + 8) - 1) > 0x3FFFFFFE
           && (unsigned int)(*(_DWORD *)(v16 + 48) - 1) <= 0x3FFFFFFE) )
        {
          return v4;
        }
LABEL_17:
        if ( v3 == v5 )
          return v3;
        v13 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v13 )
          BUG();
        v14 = *(_QWORD *)v13;
        v15 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
              break;
            v14 = *(_QWORD *)v15;
          }
        }
        v12 = *(_WORD *)(v15 + 68);
        v4 = v5;
        v5 = v15;
        if ( v12 != 20 )
          goto LABEL_15;
      }
    }
    else
    {
      if ( (unsigned __int16)(v12 - 14) <= 4u )
        goto LABEL_17;
      switch ( v12 )
      {
        case 0x49u:
        case 0x4Au:
        case 0x4Cu:
        case 0x4Du:
        case 0x4Fu:
        case 0x83u:
        case 0x84u:
        case 0x89u:
        case 0x8Bu:
          goto LABEL_17;
        default:
          return v4;
      }
    }
  }
  do
  {
    v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v5 )
      BUG();
    v20 = *(_DWORD *)(v5 + 44);
    v21 = *(_QWORD *)v5;
    v22 = v20;
    if ( (*(_QWORD *)v5 & 4) != 0 )
    {
      if ( (v20 & 4) != 0 )
        goto LABEL_53;
    }
    else if ( (v20 & 4) != 0 )
    {
      while ( 1 )
      {
        v5 = v21 & 0xFFFFFFFFFFFFFFF8LL;
        v22 = *(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 44);
        if ( (v22 & 4) == 0 )
          break;
        v21 = *(_QWORD *)v5;
      }
    }
    if ( (v22 & 8) != 0 )
    {
      v23 = sub_2E88A90(v5, 128, 1);
      goto LABEL_47;
    }
LABEL_53:
    v23 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) >> 7;
LABEL_47:
    if ( v23 )
      return v4;
  }
  while ( a2[16] != *(unsigned __int16 *)(v5 + 68) );
  return v5;
}
