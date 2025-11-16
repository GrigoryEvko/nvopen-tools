// Function: sub_1651CA0
// Address: 0x1651ca0
//
void __fastcall sub_1651CA0(__int64 *a1, __int64 a2)
{
  const char *v4; // rax
  __int64 v5; // r14
  _BYTE *v6; // rax
  __int64 v7; // rax
  char v8; // al
  char v9; // dl
  char v10; // al
  char v11; // si
  const char *v12; // rax
  bool v13; // zf
  __int64 v14; // r14
  _BYTE *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19[2]; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+10h] [rbp-30h]
  char v21; // [rsp+11h] [rbp-2Fh]

  if ( sub_15E4F60(a2) && (*(_BYTE *)(a2 + 32) & 0xF) != 9 && (*(_BYTE *)(a2 + 32) & 0xF) != 0 )
  {
    v21 = 1;
    v4 = "Global is external, but doesn't have external or weak linkage!";
    goto LABEL_5;
  }
  if ( (unsigned int)sub_15E4C60(a2) > 0x20000000 )
  {
    v21 = 1;
    v12 = "huge alignment values are unsupported";
    goto LABEL_22;
  }
  v8 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( v8 == 6 )
  {
    if ( *(_BYTE *)(a2 + 16) != 3 )
    {
      v21 = 1;
      v12 = "Only global variables can have appending linkage!";
      goto LABEL_22;
    }
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 24) + 8LL) != 14 )
    {
      v21 = 1;
      v12 = "Only global arrays can have appending linkage!";
      goto LABEL_22;
    }
    goto LABEL_14;
  }
  if ( v8 != 1 )
  {
LABEL_14:
    if ( !sub_15E4F60(a2) )
      goto LABEL_15;
  }
  if ( !sub_15E4F10(a2) )
  {
LABEL_15:
    v9 = *(_BYTE *)(a2 + 33);
    if ( (v9 & 3) != 1 )
    {
      v10 = *(_BYTE *)(a2 + 32);
      v11 = v10 & 0xF;
      if ( (v10 & 0xFu) - 7 <= 1 )
      {
        if ( (v9 & 0x40) != 0 )
        {
LABEL_18:
          v19[0] = a2;
          v19[1] = (__int64)a1;
          sub_1650C20(a2, (__int64)(a1 + 145), (__int64)sub_1650890, v19);
          return;
        }
        v21 = 1;
        v4 = "GlobalValue with private or internal linkage must be dso_local!";
LABEL_5:
        v5 = *a1;
        v19[0] = (__int64)v4;
        v20 = 3;
        if ( v5 )
        {
          sub_16E2CE0(v19, v5);
          v6 = *(_BYTE **)(v5 + 24);
          if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 16) )
          {
            sub_16E7DE0(v5, 10);
          }
          else
          {
            *(_QWORD *)(v5 + 24) = v6 + 1;
            *v6 = 10;
          }
          v7 = *a1;
          *((_BYTE *)a1 + 72) = 1;
          if ( v7 )
            goto LABEL_9;
          return;
        }
LABEL_26:
        *((_BYTE *)a1 + 72) = 1;
        return;
      }
      goto LABEL_28;
    }
    if ( (*(_BYTE *)(a2 + 33) & 0x40) != 0 )
    {
      v21 = 1;
      v4 = "GlobalValue with DLLImport Storage is dso_local!";
      goto LABEL_5;
    }
    v13 = !sub_15E4F60(a2);
    v10 = *(_BYTE *)(a2 + 32);
    if ( v13 )
    {
      v11 = v10 & 0xF;
    }
    else
    {
      v11 = v10 & 0xF;
      if ( (v10 & 0xF) == 0 )
        goto LABEL_28;
    }
    if ( v11 == 1 )
    {
LABEL_28:
      if ( (v10 & 0x30) == 0 || v11 == 9 || (*(_BYTE *)(a2 + 33) & 0x40) != 0 )
        goto LABEL_18;
      v21 = 1;
      v4 = "GlobalValue with non default visibility must be dso_local!";
      goto LABEL_5;
    }
    v14 = *a1;
    v21 = 1;
    v19[0] = (__int64)"Global is marked as dllimport, but not external";
    v20 = 3;
    if ( !v14 )
      goto LABEL_26;
    sub_16E2CE0(v19, v14);
    v15 = *(_BYTE **)(v14 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
    {
      sub_16E7DE0(v14, 10);
    }
    else
    {
      *(_QWORD *)(v14 + 24) = v15 + 1;
      *v15 = 10;
    }
    v16 = *a1;
    *((_BYTE *)a1 + 72) = 1;
    if ( !v16 )
      return;
    if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)a2, v16, 1, (__int64)(a1 + 2));
      v17 = *a1;
      v18 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v18 < *(_QWORD *)(*a1 + 16) )
        goto LABEL_43;
    }
    else
    {
      sub_155BD40(a2, v16, (__int64)(a1 + 2), 0);
      v17 = *a1;
      v18 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v18 < *(_QWORD *)(*a1 + 16) )
      {
LABEL_43:
        *(_QWORD *)(v17 + 24) = v18 + 1;
        *v18 = 10;
        return;
      }
    }
    sub_16E7DE0(v17, 10);
    return;
  }
  v21 = 1;
  v12 = "Declaration may not be in a Comdat!";
LABEL_22:
  v19[0] = (__int64)v12;
  v20 = 3;
  sub_164FF40(a1, (__int64)v19);
  if ( *a1 )
LABEL_9:
    sub_164FA80(a1, a2);
}
