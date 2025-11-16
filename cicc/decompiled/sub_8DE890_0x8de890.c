// Function: sub_8DE890
// Address: 0x8de890
//
__int64 __fastcall sub_8DE890(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v5; // r9
  __int64 v6; // r10
  unsigned __int8 v7; // si
  bool v8; // dl
  _BOOL4 v9; // edi
  __int64 result; // rax
  char v11; // r13
  char v12; // r11
  __int64 v13; // r15
  __int64 v14; // r13
  bool v15; // dl
  bool v16; // al
  __int64 v17; // r8
  __int64 v18; // r12
  _QWORD *v19; // r14
  char v20; // al
  char v21; // dl
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  int v25; // eax
  char v26; // al
  char v27; // dl
  int v28; // eax
  char v29; // dl
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned int v33; // eax
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  char v36; // [rsp+10h] [rbp-50h]

  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    a1 = *(_QWORD *)(a1 + 160);
  while ( *(_BYTE *)(a1 + 140) == 12 );
  while ( *(_BYTE *)(a2 + 140) == 12 )
  {
    a2 = *(_QWORD *)(a2 + 160);
LABEL_5:
    ;
  }
  v5 = *(_QWORD *)(a1 + 168);
  v6 = *(_QWORD *)(a2 + 168);
  v7 = *(_BYTE *)(v6 + 16);
  v8 = (*(_BYTE *)(v5 + 16) & 2) != 0;
  v9 = v8;
  if ( ((v7 ^ *(_BYTE *)(v5 + 16)) & 1) == 0 )
    goto LABEL_12;
  if ( !dword_4F077C0 )
    return 0;
  v11 = v7 & 1;
  if ( (*(_BYTE *)(v5 + 16) & 1) != 0 )
  {
    v12 = 1;
    if ( v11 )
      goto LABEL_12;
  }
  else
  {
    v26 = 1;
    v12 = (*(_BYTE *)(v5 + 16) & 2) == 0;
    if ( v11 )
      goto LABEL_52;
  }
  v26 = (*(_BYTE *)(v6 + 16) & 2) == 0;
LABEL_52:
  if ( v12 != v26 )
    return 0;
LABEL_12:
  if ( dword_4F077C4 == 2 || (result = 1, (*(_BYTE *)(v6 + 16) & 2) != 0 || v8) )
  {
    v13 = *(_QWORD *)v5;
    v14 = *(_QWORD *)v6;
    if ( (*(_BYTE *)(v6 + 16) & 2) != 0 )
    {
      if ( (*(_BYTE *)(v5 + 16) & 2) == 0 )
      {
        v13 = *(_QWORD *)v6;
        if ( (*(_BYTE *)(v5 + 16) & 4) != 0 )
          v14 = *(_QWORD *)v5;
      }
    }
    else
    {
      v9 = 0;
      if ( (*(_BYTE *)(v6 + 16) & 4) == 0 )
        v14 = *(_QWORD *)v5;
    }
    if ( (a3 & 0x200000) != 0 )
    {
      if ( v13 && (*(_BYTE *)(v13 + 35) & 1) != 0 )
      {
        if ( !v14 || (*(_BYTE *)(v14 + 35) & 1) == 0 )
        {
          v13 = *(_QWORD *)v13;
          goto LABEL_23;
        }
        while ( 1 )
        {
LABEL_25:
          if ( ((*(_BYTE *)(v13 + 33) & 3) == 1) != ((*(_BYTE *)(v14 + 33) & 3) == 1) )
            return 0;
          v17 = a3 & 0x20;
          if ( (a3 & 0x20) != 0 )
          {
            if ( ((*(_DWORD *)(v14 + 32) ^ *(_DWORD *)(v13 + 32)) & 0x3F800) != 0 )
              return 0;
          }
          else if ( unk_4D044D0
                 && (((unsigned __int8)(*(_DWORD *)(v13 + 32) >> 11)
                    ^ (unsigned __int8)(*(_DWORD *)(v14 + 32) >> 11))
                   & 8) != 0 )
          {
            return 0;
          }
          v18 = *(_QWORD *)(v13 + 8);
          v19 = *(_QWORD **)(v14 + 8);
          if ( (a3 & 0x1000) != 0 )
          {
            if ( dword_4F077C4 != 2 )
              break;
            v35 = sub_8D72A0(v13);
            v34 = sub_8D72A0(v14);
            if ( sub_8D3410(v35) && sub_8D3410(v34) )
            {
              v19 = (_QWORD *)v34;
              v18 = v35;
            }
          }
          if ( dword_4F077C4 != 2 )
            break;
LABEL_39:
          if ( !(unsigned int)sub_8DD8B0(v18, (__int64)v19, a3, a4, v17) )
          {
            if ( !dword_4F077C0 )
              return 0;
            if ( (unsigned int)sub_8D3B40(v18) )
            {
              v25 = sub_8DAB90(v18, (__int64)v19, v22, v23, v24);
            }
            else
            {
              if ( !(unsigned int)sub_8D3B40((__int64)v19) )
                return 0;
              v25 = sub_8DAB90((__int64)v19, v18, v30, v31, v32);
            }
            if ( !v25 )
              return 0;
          }
          v13 = *(_QWORD *)v13;
          v14 = *(_QWORD *)v14;
          v15 = v13 == 0;
          v16 = v14 == 0;
          if ( !v13 || !v14 )
            return v15 & (unsigned __int8)v16;
        }
        v20 = *(_BYTE *)(v18 + 140);
        if ( !unk_4D044D0 )
          goto LABEL_57;
        if ( (v20 & 0xFB) == 8 )
        {
          v27 = (unsigned int)sub_8D4C10(v18, 1) >> 3;
          LOBYTE(v28) = 0;
          v29 = v27 & 1;
          if ( (*((_BYTE *)v19 + 140) & 0xFB) != 8 )
          {
LABEL_55:
            if ( v29 != (_BYTE)v28 )
              return 0;
            v20 = *(_BYTE *)(v18 + 140);
LABEL_57:
            v21 = *((_BYTE *)v19 + 140);
            goto LABEL_33;
          }
        }
        else
        {
          v21 = *((_BYTE *)v19 + 140);
          if ( (v21 & 0xFB) != 8 )
          {
LABEL_33:
            if ( v20 == 12 )
            {
              do
                v18 = *(_QWORD *)(v18 + 160);
              while ( *(_BYTE *)(v18 + 140) == 12 );
            }
            if ( v21 == 12 )
            {
              do
                v19 = (_QWORD *)v19[20];
              while ( *((_BYTE *)v19 + 140) == 12 );
            }
            if ( !(v9 | a3 & 0x200) )
              v19 = sub_8D6740((__int64)v19);
            goto LABEL_39;
          }
          v29 = 0;
        }
        v36 = v29;
        v33 = sub_8D4C10((__int64)v19, dword_4F077C4 != 2);
        v29 = v36;
        v28 = (v33 >> 3) & 1;
        goto LABEL_55;
      }
      if ( !v14 )
      {
        v16 = 1;
        v15 = v13 == 0;
        return v15 & (unsigned __int8)v16;
      }
      if ( (*(_BYTE *)(v14 + 35) & 1) != 0 )
        v14 = *(_QWORD *)v14;
    }
LABEL_23:
    v15 = v13 == 0;
    v16 = v14 == 0;
    if ( v13 && v14 )
      goto LABEL_25;
    return v15 & (unsigned __int8)v16;
  }
  return result;
}
