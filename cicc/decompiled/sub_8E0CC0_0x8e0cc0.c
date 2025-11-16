// Function: sub_8E0CC0
// Address: 0x8e0cc0
//
__int64 __fastcall sub_8E0CC0(__int64 a1, int a2, unsigned int a3, __int64 a4, __int64 a5, int a6, __int64 a7)
{
  __int64 v8; // r13
  __int64 v9; // r12
  char v11; // al
  unsigned int v12; // r8d
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  _QWORD *v18; // rax
  __int64 v19; // r12
  __int64 v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  int v24; // ebx
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // [rsp+10h] [rbp-50h]
  int v29; // [rsp+28h] [rbp-38h] BYREF
  int v30[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v8 = a5;
  v9 = a1;
  *(_OWORD *)a7 = 0;
  *(_QWORD *)(a7 + 16) = 0;
  *(_BYTE *)(a7 + 12) |= 0x20u;
  v11 = *(_BYTE *)(a1 + 140);
  if ( v11 != 12 )
    goto LABEL_5;
  do
  {
    v9 = *(_QWORD *)(v9 + 160);
    v11 = *(_BYTE *)(v9 + 140);
  }
  while ( v11 == 12 );
  if ( *(_BYTE *)(a5 + 140) == 12 )
  {
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(v8 + 140) == 12 );
  }
  if ( v11 != 13 )
  {
    if ( !sub_8D3D40(v9) )
    {
      if ( !a2 || !sub_712690(a4) )
        return (*(_BYTE *)(v9 + 140) == 19) | (unsigned __int8)(*(_BYTE *)(v9 + 140) == 0);
      if ( *(_BYTE *)(v9 + 140) != 19 )
        *(_BYTE *)(a7 + 12) |= 0x18u;
    }
    return 1;
  }
  v14 = sub_8D4890(v9);
  v15 = sub_8D4890(v8);
  if ( v14 == v15 || (v27 = v15, (unsigned int)sub_8D97D0(v14, v15, 0, v16, v17)) )
  {
    *(_BYTE *)(a7 + 12) &= ~0x20u;
  }
  else
  {
    v18 = sub_8D5CE0(v27, v14);
    v12 = 0;
    if ( v18 )
    {
      *(_BYTE *)(a7 + 12) |= 1u;
      *(_QWORD *)a7 = v18;
    }
    else if ( (*(_BYTE *)(v14 + 177) & 0x20) == 0 && (*(_BYTE *)(v27 + 177) & 0x20) == 0 )
    {
      return v12;
    }
  }
  v19 = sub_8D4870(v9);
  v20 = sub_8D4870(v8);
  v12 = sub_8E0BF0(v20, v19, a3, a6, v30);
  if ( v12 )
  {
    *(_BYTE *)(a7 + 12) = (2 * (v30[0] & 1)) | *(_BYTE *)(a7 + 12) & 0xFD;
    if ( !a6 )
    {
      if ( sub_8D2310(v20) && !(unsigned int)sub_8DBCE0(v19, v20, v21, v22, v23) )
        *(_BYTE *)(a7 + 13) |= 4u;
      if ( (*(_BYTE *)(v20 + 140) & 0xFB) == 8 )
      {
        v24 = sub_8D4C10(v20, dword_4F077C4 != 2);
        v25 = 0;
        if ( (*(_BYTE *)(v19 + 140) & 0xFB) != 8 )
          goto LABEL_24;
        goto LABEL_23;
      }
      if ( (*(_BYTE *)(v19 + 140) & 0xFB) == 8 )
      {
        v24 = 0;
LABEL_23:
        v25 = sub_8D4C10(v19, dword_4F077C4 != 2);
LABEL_24:
        if ( v25 != v24 && (unsigned int)sub_8DF7B0(v19, v20, v30, &v29, 0) )
        {
          v12 = 1;
          *(_BYTE *)(a7 + 12) = (2 * (v30[0] & 1)) | *(_BYTE *)(a7 + 12) & 0xFD;
          *(_DWORD *)(a7 + 8) = v29;
          return v12;
        }
      }
    }
    return 1;
  }
  if ( dword_4F04C44 != -1
    || (v26 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v26 + 6) & 6) != 0)
    || *(_BYTE *)(v26 + 4) == 12 )
  {
    if ( !(unsigned int)sub_8DBE70(v19) )
      return (unsigned int)sub_8DBE70(v20) != 0;
    return 1;
  }
  return v12;
}
