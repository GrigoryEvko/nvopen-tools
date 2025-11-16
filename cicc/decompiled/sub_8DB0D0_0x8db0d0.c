// Function: sub_8DB0D0
// Address: 0x8db0d0
//
_BOOL8 __fastcall sub_8DB0D0(__int64 a1, __int64 a2, _DWORD *a3, _DWORD *a4, __int64 a5)
{
  __int64 v7; // rbx
  char v8; // r14
  char v9; // al
  char i; // dl
  char v11; // al
  int v12; // eax
  _BOOL4 v13; // r13d
  char v15; // al
  char v16; // al
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  _BOOL4 v19; // r13d
  _BOOL4 v20; // eax
  int v21; // eax
  __int64 v22; // rax
  _BOOL4 v23; // r13d
  __int64 v24; // rax
  __int64 j; // r15
  __int64 v26; // rax
  char k; // dl
  char v28; // al
  __int64 m; // rdx
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  _DWORD *v32; // [rsp+8h] [rbp-38h]
  _DWORD *v33; // [rsp+8h] [rbp-38h]

  v7 = a2;
  *a3 = 0;
  *a4 = 0;
  v8 = dword_4D04964;
  if ( dword_4D04964 )
    v8 = byte_4F07472[0] == 8;
  while ( 1 )
  {
    v9 = *(_BYTE *)(a1 + 140);
    if ( v9 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( (*(_BYTE *)(v7 + 141) & 0x20) != 0 )
    goto LABEL_19;
  if ( v9 == 6 )
  {
    if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
    {
      v11 = *(_BYTE *)(a1 + 140);
      if ( v11 == 6 )
        goto LABEL_16;
      goto LABEL_14;
    }
  }
  else
  {
    if ( v9 != 19 )
    {
      if ( v9 == 2 )
      {
        if ( i == 6 && (*(_BYTE *)(v7 + 168) & 1) == 0
          || dword_4F077C4 == 2
          && (unk_4F07778 > 201102 || dword_4F07774)
          && ((v32 = a4, v7 == a1) || (v21 = sub_8D97D0(a1, v7, 0, (__int64)a4, a5), a4 = v32, v21)) )
        {
          if ( (unsigned int)sub_8D2E30(v7) )
          {
            while ( *(_BYTE *)(a1 + 140) == 12 )
              a1 = *(_QWORD *)(a1 + 160);
            while ( *(_BYTE *)(v7 + 140) == 12 )
              v7 = *(_QWORD *)(v7 + 160);
            if ( *(_QWORD *)(v7 + 128) < *(_QWORD *)(a1 + 128) )
            {
              *a3 = 1053;
              return 1;
            }
          }
          return 1;
        }
      }
      goto LABEL_13;
    }
    if ( (*(_BYTE *)(a1 + 141) & 0x20) != 0 )
      goto LABEL_13;
  }
  if ( i == 2 && (unk_4D04000 || (*(_BYTE *)(v7 + 161) & 8) == 0) )
  {
    v17 = *(_QWORD *)(v7 + 128);
    v18 = *(_QWORD *)(a1 + 128);
    if ( dword_4F077C4 != 2 || dword_4F077BC )
    {
      if ( v17 < v18 )
      {
        *a3 = 767;
        v13 = 1;
        *a4 = 1;
        return v13;
      }
    }
    else if ( v17 < v18 )
    {
      goto LABEL_13;
    }
    if ( v17 != v18 )
      return 1;
    *a3 = 1375;
LABEL_83:
    *a4 = 1;
    return 1;
  }
LABEL_13:
  v11 = *(_BYTE *)(a1 + 140);
  if ( v11 != 6 )
  {
LABEL_14:
    if ( v11 != 13 || *(_BYTE *)(v7 + 140) != 13 )
      goto LABEL_16;
    v22 = sub_8D4870(a1);
    v23 = sub_8D2310(v22);
    v24 = sub_8D4870(v7);
    if ( v23 != sub_8D2310(v24) )
      goto LABEL_19;
    return 1;
  }
  if ( (*(_BYTE *)(a1 + 168) & 1) == 0 && *(_BYTE *)(v7 + 140) == 6 )
  {
    if ( (*(_BYTE *)(v7 + 168) & 1) != 0 )
    {
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_32;
      goto LABEL_17;
    }
    v33 = a4;
    for ( j = sub_8D46C0(a1); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v26 = sub_8D46C0(v7);
    a4 = v33;
    for ( k = *(_BYTE *)(v26 + 140); k == 12; k = *(_BYTE *)(v26 + 140) )
      v26 = *(_QWORD *)(v26 + 160);
    v28 = *(_BYTE *)(j + 140);
    if ( k == 14 || v28 == 14 || (v28 == 7) == (k == 7) )
      return 1;
    LOBYTE(v12) = *(_BYTE *)(v7 + 140);
    if ( dword_4F077C4 != 2 && (v8 & 1) != 0 )
      goto LABEL_20;
    for ( m = a1; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
      ;
    v30 = *(_QWORD *)(m + 128);
    if ( (_BYTE)v12 == 12 )
    {
      v31 = v7;
      do
        v31 = *(_QWORD *)(v31 + 160);
      while ( *(_BYTE *)(v31 + 140) == 12 );
      if ( v30 > *(_QWORD *)(v31 + 128) )
        goto LABEL_32;
    }
    else if ( v30 > *(_QWORD *)(v7 + 128) )
    {
      goto LABEL_20;
    }
    if ( dword_4F077C4 == 2 )
      return 1;
    if ( !dword_4D04964 )
      return 1;
    *a3 = 1235;
    if ( byte_4F07472[0] > 7u )
      return 1;
    goto LABEL_83;
  }
LABEL_16:
  if ( !HIDWORD(qword_4F077B4) )
    goto LABEL_19;
LABEL_17:
  if ( !sub_8D2B80(a1) && !sub_8D2B80(v7) || *(_QWORD *)(a1 + 128) != *(_QWORD *)(v7 + 128) )
    goto LABEL_19;
  v19 = sub_8D2B80(a1);
  v20 = sub_8D2B80(v7);
  if ( !v19 )
  {
    if ( v20 && *(_BYTE *)(a1 + 140) == 2 )
      return 1;
LABEL_19:
    LOBYTE(v12) = *(_BYTE *)(v7 + 140);
    goto LABEL_20;
  }
  if ( v20 )
    return 1;
  v12 = *(unsigned __int8 *)(v7 + 140);
  if ( (_BYTE)v12 == 2 )
    return 1;
LABEL_20:
  if ( !(_BYTE)v12 )
  {
LABEL_21:
    if ( *(_BYTE *)(a1 + 140) )
    {
      if ( !sub_8D3D40(a1) )
      {
        v15 = *(_BYTE *)(a1 + 140);
        if ( v15 != 2 && (v15 != 6 || (*(_BYTE *)(a1 + 168) & 1) != 0) )
          return v15 == 13;
      }
    }
    return 1;
  }
LABEL_32:
  v13 = sub_8D3D40(v7);
  if ( v13 )
    goto LABEL_21;
  if ( !*(_BYTE *)(a1 + 140) || sub_8D3D40(a1) )
  {
    v16 = *(_BYTE *)(v7 + 140);
    if ( v16 != 2 )
    {
      if ( v16 == 6 )
      {
        if ( (*(_BYTE *)(v7 + 168) & 1) != 0 )
          return v13;
      }
      else if ( v16 != 13 )
      {
        return v13;
      }
    }
    return 1;
  }
  return v13;
}
