// Function: sub_8187D0
// Address: 0x8187d0
//
void __fastcall sub_8187D0(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4)
{
  __int64 v4; // r9
  unsigned __int8 v7; // al
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // rbx
  char v11; // dl
  char v12; // dl
  char v13; // cl
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-38h]

  v4 = a2;
  v7 = *(_BYTE *)(a1 + 56);
  v8 = *(_QWORD *)(a1 + 72);
  if ( v7 <= 0x65u )
  {
    if ( v7 <= 0x63u )
    {
      if ( v7 <= 0x17u )
      {
        if ( v7 <= 0x15u )
          goto LABEL_59;
        v9 = *(_QWORD *)(v8 + 16);
        v10 = 0;
        if ( v9 )
        {
          v11 = *(_BYTE *)(v9 + 24);
          if ( v11 == 2 || v11 == 3 || v11 == 20 || v11 == 4 || v11 == 22 )
          {
            v10 = *(_QWORD *)(v9 + 64);
            v9 = 0;
            if ( (*(_BYTE *)(a1 + 27) & 2) == 0 )
              goto LABEL_13;
            goto LABEL_33;
          }
          v10 = 0;
          v9 = 0;
        }
LABEL_12:
        if ( (*(_BYTE *)(a1 + 27) & 2) == 0 )
          goto LABEL_13;
LABEL_33:
        if ( (*(_BYTE *)(a1 + 26) & 2) == 0 )
        {
          if ( *(_BYTE *)(v8 + 24) != 24 )
            goto LABEL_13;
          if ( *(_DWORD *)(v8 + 56) )
          {
            v14 = *(_QWORD *)(a1 + 72);
            v8 = v9;
LABEL_48:
            if ( v7 > 0x6Du )
              goto LABEL_59;
            v9 = v8;
            v8 = v14;
LABEL_13:
            if ( v7 > 0x5Du )
              goto LABEL_14;
            if ( v7 == 22 )
            {
LABEL_40:
              v15 = qword_4F18BE0;
              *a4 += 2;
              sub_8238B0(v15, "dt", 2);
              v16 = a2;
LABEL_41:
              v18 = v16;
              sub_816460(v8, a3, 0, a4);
              v4 = v18;
              if ( !v9 )
              {
LABEL_42:
                sub_8129A0(*(_QWORD *)v8, v10, a4);
                return;
              }
              v14 = v8;
              v8 = v9;
LABEL_36:
              sub_817D30(v8, v4, v14, a3, a4);
              return;
            }
            if ( v7 == 23 )
            {
LABEL_44:
              v17 = qword_4F18BE0;
              *a4 += 2;
              sub_8238B0(v17, "pt", 2);
              v16 = a2;
              goto LABEL_41;
            }
LABEL_59:
            sub_721090();
          }
        }
        if ( v9 )
        {
LABEL_35:
          v8 = v9;
          v14 = 0;
          goto LABEL_36;
        }
        goto LABEL_66;
      }
      if ( (unsigned __int8)(v7 - 94) > 1u )
        goto LABEL_59;
    }
    v9 = *(_QWORD *)(v8 + 16);
    v12 = *(_BYTE *)(v9 + 24);
    if ( v12 == 2 || v12 == 3 || v12 == 20 || v12 == 4 || (v10 = 0, v12 == 22) )
    {
      v10 = *(_QWORD *)(v9 + 64);
      if ( (*(_BYTE *)(a1 + 27) & 2) == 0 )
        goto LABEL_13;
      goto LABEL_33;
    }
    goto LABEL_12;
  }
  if ( (unsigned __int8)(v7 - 106) > 3u )
    goto LABEL_59;
  v13 = *(_BYTE *)(v8 + 24);
  v14 = *(_QWORD *)(v8 + 16);
  if ( v13 == 2 || v13 == 3 || v13 == 20 || v13 == 4 || (v10 = 0, v13 == 22) )
    v10 = *(_QWORD *)(v8 + 64);
  if ( (*(_BYTE *)(a1 + 27) & 2) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 26) & 2) != 0 )
    {
      v14 = 0;
      goto LABEL_36;
    }
    if ( v13 != 24 )
    {
      if ( !v14 )
        goto LABEL_36;
      goto LABEL_48;
    }
    if ( !*(_DWORD *)(v8 + 56) )
    {
      v9 = *(_QWORD *)(a1 + 72);
      goto LABEL_35;
    }
    if ( v14 )
      goto LABEL_48;
    if ( v8 )
      goto LABEL_36;
LABEL_66:
    BUG();
  }
  if ( !v14 )
    goto LABEL_36;
  v9 = *(_QWORD *)(a1 + 72);
  v8 = *(_QWORD *)(v8 + 16);
LABEL_14:
  switch ( v7 )
  {
    case '^':
    case 'd':
    case 'j':
      goto LABEL_40;
    case '_':
    case 'e':
    case 'k':
      goto LABEL_44;
    case 'l':
      *a4 += 2;
      sub_8238B0(qword_4F18BE0, "ds", 2);
      goto LABEL_53;
    case 'm':
      *a4 += 2;
      sub_8238B0(qword_4F18BE0, "pm", 2);
LABEL_53:
      sub_816460(v8, a3, 0, a4);
      if ( !v9 )
        goto LABEL_42;
      sub_816460(v9, a3, 0, a4);
      break;
    default:
      goto LABEL_59;
  }
}
