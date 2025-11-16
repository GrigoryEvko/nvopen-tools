// Function: sub_5D3810
// Address: 0x5d3810
//
unsigned __int64 __fastcall sub_5D3810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rdi
  __int64 v6; // rbx
  char v7; // al
  unsigned int v8; // eax
  unsigned __int64 v9; // r13
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // [rsp-30h] [rbp-30h]
  __int64 v16; // [rsp-30h] [rbp-30h]
  __int64 v17; // [rsp-30h] [rbp-30h]

  if ( unk_4F077C4 != 2 || *(_BYTE *)(a3 + 140) == 11 )
    return 0;
  if ( (*(_BYTE *)(a2 + 144) & 4) != 0 && (!a1 || (*(_BYTE *)(a1 + 145) & 0x10) == 0) )
    return 0;
  for ( i = *(_QWORD *)(a2 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(a2 + 145) & 0x10) != 0 )
  {
    do
    {
      v6 = *(_QWORD *)(i + 160);
      for ( i = *(_QWORD *)(v6 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
    }
    while ( (*(_BYTE *)(v6 + 145) & 0x10) != 0 );
  }
  else
  {
    v6 = a2;
  }
  v7 = *(_BYTE *)(i + 140);
  if ( v7 == 8 )
  {
    v17 = a3;
    v14 = sub_8D40F0(i);
    a3 = v17;
    v7 = *(_BYTE *)(v14 + 140);
  }
  if ( (unsigned __int8)(v7 - 9) <= 2u || (*(_BYTE *)(a2 + 145) & 0x10) != 0 )
  {
    v16 = a3;
    v8 = sub_88CF10(*(_QWORD *)(v6 + 120));
    v9 = *(unsigned int *)(v6 + 140);
    if ( (_DWORD)v9 )
      goto LABEL_24;
    v10 = v16;
    v9 = 1;
    if ( (*(_BYTE *)(v6 + 144) & 1) != 0 )
      goto LABEL_24;
    goto LABEL_20;
  }
  if ( a1 && (*(_BYTE *)(a1 + 145) & 0x10) != 0 )
  {
    v15 = a3;
    v8 = sub_88CF10(*(_QWORD *)(v6 + 120));
    v9 = *(unsigned int *)(v6 + 140);
    if ( (_DWORD)v9 )
      goto LABEL_32;
    v9 = 1;
    if ( (*(_BYTE *)(v6 + 144) & 1) != 0 )
      goto LABEL_32;
    v10 = v15;
LABEL_20:
    v11 = *(unsigned int *)(v10 + 184);
    v9 = v8;
    if ( (_DWORD)v11 && (_DWORD)v11 != unk_4F072D4 && (unsigned int)v11 <= v8 )
      v9 = v11;
LABEL_24:
    v12 = 0;
    if ( !a1 )
      return *(_QWORD *)(a2 + 128) - v12;
LABEL_32:
    v12 = sub_5D35E0(a1);
    if ( v12 % v9 )
    {
      v13 = v12 + v9 - v12 % v9;
      if ( (*(_BYTE *)(v6 + 144) & 4) == 0 )
        v12 = v13;
    }
    return *(_QWORD *)(a2 + 128) - v12;
  }
  return 0;
}
