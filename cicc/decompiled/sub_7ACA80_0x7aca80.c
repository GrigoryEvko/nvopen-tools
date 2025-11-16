// Function: sub_7ACA80
// Address: 0x7aca80
//
__int64 __fastcall sub_7ACA80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, int a7, int a8)
{
  __int64 v10; // r14
  __int64 v11; // r13
  char v12; // al
  __int64 result; // rax
  char v14; // al
  char v15; // al
  int v16; // r9d
  char v17; // al
  unsigned __int8 v18; // r15
  char v19; // al
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rdi
  int v26; // eax
  char v27; // al
  __int64 v28; // rsi
  __int64 v29; // rdi
  char v30; // al
  char v31; // al
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]
  int v40; // [rsp+8h] [rbp-38h]

  if ( !a3 )
    goto LABEL_7;
  v10 = a2;
  v11 = a3;
  if ( !a5 )
    goto LABEL_8;
  if ( !a7 )
  {
    if ( a2 == a4 )
      goto LABEL_8;
    v14 = *(_BYTE *)(a2 + 80);
    if ( (unsigned __int8)(v14 - 4) <= 1u )
      goto LABEL_28;
    if ( v14 != 3 )
      goto LABEL_13;
    v25 = *(_QWORD *)(a2 + 88);
LABEL_57:
    v40 = a6;
    v26 = sub_8D3A70(v25);
    a6 = v40;
    if ( !v26 )
    {
      v14 = *(_BYTE *)(a2 + 80);
LABEL_13:
      if ( v14 != 19 )
        goto LABEL_14;
      goto LABEL_34;
    }
LABEL_28:
    if ( *(_BYTE *)(a4 + 80) == 3 && *(_BYTE *)(a4 + 104) )
    {
      v21 = *(_QWORD *)(a2 + 88);
      v22 = *(_QWORD *)(a4 + 88);
      if ( v22 == v21 )
        goto LABEL_8;
      goto LABEL_31;
    }
LABEL_14:
    v38 = a6;
    v15 = sub_877F80(a4);
    v16 = v38;
    if ( v15 != 1 )
      goto LABEL_15;
    v17 = *(_BYTE *)(a2 + 80);
    if ( (unsigned __int8)(v17 - 4) > 1u )
    {
      if ( v17 != 3 )
      {
        if ( (*(_BYTE *)(a4 + 84) & 2) == 0 )
          goto LABEL_17;
        goto LABEL_43;
      }
      v24 = sub_8D3A70(*(_QWORD *)(a2 + 88));
      v16 = v38;
      if ( !v24 )
      {
LABEL_15:
        if ( (*(_BYTE *)(a4 + 84) & 2) == 0 )
        {
          v17 = *(_BYTE *)(a2 + 80);
          if ( v17 == 3 )
          {
            v27 = *(_BYTE *)(a4 + 80);
            if ( v27 == 3 )
            {
              if ( dword_4F077C4 != 2 )
              {
LABEL_64:
                if ( !dword_4F077BC || qword_4F077A8 <= 0x765Bu )
                {
                  v28 = *(_QWORD *)(a2 + 88);
                  v29 = *(_QWORD *)(a4 + 88);
                  if ( v28 != v29 && !(unsigned int)sub_8D97D0(v29, v28, 0, &dword_4F077C4, dword_4F077BC) )
                    goto LABEL_18;
LABEL_8:
                  result = v10;
                  goto LABEL_9;
                }
                goto LABEL_7;
              }
LABEL_62:
              if ( unk_4F07778 > 201102 || dword_4F07774 )
                goto LABEL_7;
              goto LABEL_64;
            }
            if ( dword_4F077C4 != 2 )
            {
LABEL_18:
              v18 = byte_4F07472[0];
              v19 = sub_877F80(a4);
              v20 = a4;
              if ( v19 == 1 )
                v20 = **(_QWORD **)(a4 + 64);
              if ( a8 | a7 )
              {
                result = a4;
                v11 = a5;
                if ( a7 && !dword_4D04964 )
                  v18 = 5;
              }
              else if ( *(_BYTE *)(v10 + 80) != 23 && a5 == a4 )
              {
                result = a4;
                v11 = a5;
              }
              else
              {
                result = v10;
                v10 = v20;
                v20 = result;
              }
              if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
              {
                v39 = result;
                sub_686B60(v18, 0x390u, (FILE *)dword_4F07508, v20, v10);
                result = v39;
              }
              goto LABEL_9;
            }
LABEL_61:
            if ( (unsigned __int8)(v27 - 4) <= 2u )
              goto LABEL_62;
            goto LABEL_18;
          }
LABEL_17:
          if ( dword_4F077C4 != 2 || (unsigned __int8)(v17 - 4) > 2u )
            goto LABEL_18;
          v27 = *(_BYTE *)(a4 + 80);
          if ( v27 == 3 )
            goto LABEL_62;
          goto LABEL_61;
        }
LABEL_43:
        if ( !v16 || (unsigned __int8)(*(_BYTE *)(a4 + 80) - 19) > 3u )
          goto LABEL_8;
        goto LABEL_7;
      }
    }
    v21 = *(_QWORD *)(a2 + 88);
    v22 = *(_QWORD *)(a4 + 64);
    if ( v22 == v21 )
      goto LABEL_8;
LABEL_31:
    if ( (unsigned int)sub_8D97D0(v21, v22, 0, a4, a5) )
      goto LABEL_8;
    goto LABEL_18;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a4 + 80) - 19) <= 3u )
  {
    v12 = *(_BYTE *)(a2 + 80);
    if ( v12 == 19 )
    {
      if ( a2 == a4 )
        goto LABEL_8;
LABEL_34:
      if ( *(_BYTE *)(a4 + 80) == 3 )
      {
        if ( *(_BYTE *)(a4 + 104) )
        {
          v23 = *(_QWORD *)(a4 + 88);
          if ( (*(_BYTE *)(v23 + 177) & 0x10) != 0 )
          {
            if ( *(_QWORD *)(*(_QWORD *)(v23 + 168) + 168LL) )
            {
              if ( v11 == sub_880FE0(a4) )
                goto LABEL_7;
              goto LABEL_18;
            }
          }
        }
      }
      goto LABEL_14;
    }
    if ( v12 != 3 )
      goto LABEL_7;
    if ( !*(_BYTE *)(a2 + 104) )
      goto LABEL_7;
    v25 = *(_QWORD *)(a2 + 88);
    if ( (*(_BYTE *)(v25 + 177) & 0x10) == 0 || !*(_QWORD *)(*(_QWORD *)(v25 + 168) + 168LL) )
      goto LABEL_7;
    if ( a2 == a4 )
    {
      result = a4;
      goto LABEL_9;
    }
    goto LABEL_57;
  }
  if ( (unsigned __int8)sub_877F80(a5) == 1 )
    goto LABEL_8;
  if ( (*(_BYTE *)(a4 + 84) & 2) == 0 || (unsigned __int8)(*(_BYTE *)(a4 + 80) - 19) <= 3u )
    goto LABEL_7;
  v30 = *(_BYTE *)(a2 + 80);
  if ( v30 == 19 )
    goto LABEL_8;
  if ( v30 == 3 )
  {
    if ( !*(_BYTE *)(a2 + 104) )
      goto LABEL_116;
    v34 = *(_QWORD *)(a2 + 88);
    if ( (*(_BYTE *)(v34 + 177) & 0x10) != 0 && *(_QWORD *)(*(_QWORD *)(v34 + 168) + 168LL) )
      goto LABEL_8;
    if ( (*(_BYTE *)(v34 + 177) & 0x10) == 0 || !*(_QWORD *)(*(_QWORD *)(v34 + 168) + 168LL) )
      goto LABEL_116;
  }
  else
  {
    if ( (unsigned __int8)(v30 - 20) <= 1u )
      goto LABEL_79;
    if ( ((v30 - 7) & 0xFD) != 0 )
      goto LABEL_136;
    v35 = *(_QWORD *)(a2 + 88);
    if ( !v35 )
      goto LABEL_116;
    if ( (*(_BYTE *)(v35 + 170) & 0x10) == 0 || !**(_QWORD **)(v35 + 216) )
    {
LABEL_136:
      if ( v30 != 17 || !(unsigned int)sub_8780F0(a2) )
      {
LABEL_116:
        if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
          goto LABEL_7;
LABEL_82:
        if ( qword_4F077A8 > 0x1116Fu
          || (*(_BYTE *)(v11 + 81) & 0x10) == 0
          || *(char *)(*(_QWORD *)(v11 + 64) + 177LL) >= 0 )
        {
          goto LABEL_7;
        }
        v31 = *(_BYTE *)(v11 + 80);
        if ( v31 == 19 )
          goto LABEL_8;
        if ( v31 == 3 )
        {
          if ( *(_BYTE *)(v11 + 104) )
          {
            v37 = *(_QWORD *)(v11 + 88);
            if ( (*(_BYTE *)(v37 + 177) & 0x10) != 0 )
            {
              if ( *(_QWORD *)(*(_QWORD *)(v37 + 168) + 168LL) )
                goto LABEL_8;
            }
          }
        }
        else
        {
          if ( (unsigned __int8)(v31 - 20) <= 1u )
            goto LABEL_8;
          if ( ((v31 - 7) & 0xFD) == 0 )
          {
            v32 = *(_QWORD *)(v11 + 88);
            if ( !v32 )
              goto LABEL_7;
            if ( (*(_BYTE *)(v32 + 170) & 0x10) != 0 && **(_QWORD **)(v32 + 216) )
              goto LABEL_8;
          }
          if ( v31 == 17 && (unsigned int)sub_8780F0(v11) )
            goto LABEL_8;
        }
LABEL_7:
        result = a4;
        v11 = a5;
        goto LABEL_9;
      }
    }
  }
LABEL_79:
  if ( dword_4F077BC )
  {
    if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0xEA5Fu )
      goto LABEL_82;
  }
  else if ( !(_DWORD)qword_4F077B4 )
  {
    goto LABEL_7;
  }
  if ( dword_4F04C44 == -1 )
  {
    v36 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v36 + 6) & 6) == 0 && *(_BYTE *)(v36 + 4) != 12 )
      goto LABEL_8;
  }
  if ( *(_BYTE *)(a4 + 80) != 2 )
    goto LABEL_8;
  v33 = *(_QWORD *)(a4 + 88);
  if ( !v33 || *(_BYTE *)(v33 + 173) != 12 )
    goto LABEL_8;
  if ( (unk_4D04A11 & 0x40) == 0 )
  {
    unk_4D04A10 &= ~0x80u;
    unk_4D04A18 = 0;
  }
  result = sub_7D2AC0(&qword_4D04A00, a1, 0x2000);
  v11 = result;
  if ( result )
  {
    if ( *(_BYTE *)(result + 80) == 16 )
      result = **(_QWORD **)(result + 88);
    if ( *(_BYTE *)(result + 80) == 24 )
      result = *(_QWORD *)(result + 88);
  }
  else
  {
    result = 0;
  }
LABEL_9:
  unk_4D04A18 = v11;
  return result;
}
