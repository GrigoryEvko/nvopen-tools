// Function: sub_8B1370
// Address: 0x8b1370
//
void __fastcall sub_8B1370(__int64 a1, unsigned __int8 a2, __int64 *a3, unsigned int a4, int a5, int a6)
{
  __int64 v8; // r14
  __int64 v9; // r15
  char v10; // al
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  int v14; // eax
  int v15; // eax
  unsigned int v16; // eax
  int v17; // r10d
  __int64 v18; // r13
  _BOOL4 v19; // eax
  _BOOL4 v20; // eax
  int v21; // [rsp+4h] [rbp-3Ch]
  int v22; // [rsp+4h] [rbp-3Ch]
  int v23; // [rsp+4h] [rbp-3Ch]
  int v25; // [rsp+Ch] [rbp-34h]

  if ( a5 && !(dword_4F077BC | a4) )
  {
    v22 = a5;
    sub_88F6E0(a1);
    a5 = v22;
  }
  v8 = *(_QWORD *)(a1 + 88);
  *(_BYTE *)(v8 + 88) |= 4u;
  if ( a2 == 17 )
  {
    v12 = (_QWORD *)sub_823970(16);
    *v12 = 0;
    v13 = qword_4F60208;
    v12[1] = v8;
    *v12 = v13;
    qword_4F60208 = (__int64)v12;
    return;
  }
  if ( a2 == 18 )
  {
    v11 = *(_QWORD *)(v8 + 168);
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v8) && (unsigned int)sub_8D3A70(v8) )
      sub_8AD220(v8, 0);
    sub_71BC30(v8);
    *(_BYTE *)(v11 + 110) |= 0x20u;
  }
  else
  {
    if ( dword_4F077C4 == 2 )
    {
      v23 = a5;
      v14 = sub_8D23B0(v8);
      a5 = v23;
      if ( v14 )
      {
        v15 = sub_8D3A70(v8);
        a5 = v23;
        if ( v15 )
        {
          sub_8AD220(v8, 0);
          a5 = v23;
        }
      }
    }
    v21 = a5;
    if ( (unsigned int)sub_8D23B0(v8) )
    {
      if ( v21 )
      {
        v16 = sub_67F240();
        sub_685A50(v16, a3, (FILE *)v8, 8u);
      }
      return;
    }
    if ( a2 == 16 )
    {
      if ( !(a4 | a6) && (*(_BYTE *)(v8 + 178) & 8) != 0 )
      {
        sub_685440(dword_4F077BC == 0 ? 7 : 5, 0x641u, a1);
        return;
      }
      *(_BYTE *)(v8 + 178) |= 0x10u;
    }
    else if ( a2 == 15 )
    {
      if ( v21 && (*(_BYTE *)(v8 + 178) & 8) != 0 )
        sub_685440(7u, 0x2F8u, a1);
      *(_BYTE *)(v8 + 178) = *(_BYTE *)(v8 + 178) & 0xE7 | 8;
      sub_71BC30(v8);
    }
    v9 = **(_QWORD **)(a1 + 96);
    if ( v9 )
    {
      while ( 1 )
      {
        v10 = *(_BYTE *)(v9 + 80);
        if ( (*(_BYTE *)(v9 + 81) & 0x10) == 0 )
          goto LABEL_17;
        if ( v10 == 10 )
          goto LABEL_37;
        if ( v10 != 17 )
          break;
        v18 = *(_QWORD *)(v9 + 88);
        if ( v18 )
        {
          v10 = *(_BYTE *)(v18 + 80);
          v17 = 1;
          goto LABEL_38;
        }
LABEL_20:
        v9 = *(_QWORD *)(v9 + 16);
        if ( !v9 )
          return;
      }
      if ( v10 != 20 )
      {
LABEL_17:
        if ( v10 == 9 )
        {
          if ( sub_891C80(v9, 0, a4, a2) && !unk_4D03FD8 && sub_891C80(v9, 1, a4, a2) )
            sub_8ACB90(v9, a2, a3, 1, a4, a6, 0);
        }
        else if ( (unsigned __int8)(v10 - 4) <= 1u )
        {
          sub_8B1370(v9, a2, a3, a4, 0, 0);
        }
        goto LABEL_20;
      }
LABEL_37:
      v17 = 0;
      v18 = v9;
LABEL_38:
      if ( (unsigned __int8)(v10 - 10) <= 1u )
        goto LABEL_43;
      while ( 1 )
      {
        if ( v10 == 17 )
          goto LABEL_43;
        while ( 1 )
        {
          if ( !v17 )
            goto LABEL_20;
          v18 = *(_QWORD *)(v18 + 8);
          if ( !v18 )
            goto LABEL_20;
          v10 = *(_BYTE *)(v18 + 80);
          if ( (unsigned __int8)(v10 - 10) > 1u )
            break;
LABEL_43:
          v25 = v17;
          v19 = sub_891C80(v18, 0, a4, a2);
          v17 = v25;
          if ( v19 && !unk_4D03FD8 )
          {
            v20 = sub_891C80(v18, 1, a4, a2);
            v17 = v25;
            if ( v20 )
            {
              sub_8ACB90(v18, a2, a3, 1, a4, a6, 0);
              v17 = v25;
            }
          }
        }
      }
    }
  }
}
