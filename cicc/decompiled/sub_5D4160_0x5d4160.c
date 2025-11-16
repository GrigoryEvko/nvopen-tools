// Function: sub_5D4160
// Address: 0x5d4160
//
int __fastcall sub_5D4160(__int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // rdi
  __int64 i; // rax
  __int64 v5; // rax
  char v6; // bl
  const char *v7; // r13
  char v8; // al
  int v9; // r15d
  bool v10; // zf
  char *v11; // rax
  int v12; // edi
  char *v13; // rbx
  int v14; // r15d
  const char *v15; // r12
  int result; // eax
  int v17; // r14d

  v2 = *(_QWORD *)(a1 + 120);
  v3 = 0;
  if ( (*(_BYTE *)(v2 + 140) & 0xFB) == 8 )
    v3 = sub_8D4C10(v2, unk_4F077C4 != 2) & 0xFFFFFFFELL;
  sub_746940(v3, -1, 1, &qword_4CF7CE0);
  for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = sub_7462A0(*(unsigned __int8 *)(i + 160), 1);
  v6 = *(_BYTE *)v5;
  v7 = (const char *)v5;
  if ( *(_BYTE *)v5 == 115 )
  {
    if ( *(_BYTE *)(v5 + 1) == 105 && *(_BYTE *)(v5 + 2) == 103 )
      goto LABEL_17;
  }
  else if ( v6 == 117 && *(_BYTE *)(v5 + 1) == 110 && *(_BYTE *)(v5 + 2) == 115 )
  {
LABEL_17:
    v15 = (const char *)(v5 + 1);
    v17 = strlen((const char *)v5);
    do
    {
LABEL_12:
      ++v15;
      result = putc(v6, stream);
      v6 = *(v15 - 1);
    }
    while ( v6 );
    v14 = dword_4CF7F40;
    goto LABEL_14;
  }
  v8 = *(_BYTE *)(a1 + 144) & 8;
  v9 = v8 == 0 ? 9 : 7;
  v10 = v8 == 0;
  v11 = "signed ";
  if ( v10 )
    v11 = "unsigned ";
  v12 = *v11;
  v13 = v11 + 1;
  do
  {
    ++v13;
    putc(v12, stream);
    v12 = *(v13 - 1);
  }
  while ( *(v13 - 1) );
  v14 = dword_4CF7F40 + v9;
  v15 = v7 + 1;
  dword_4CF7F40 = v14;
  result = strlen(v7);
  v6 = *v7;
  v17 = result;
  if ( *v7 )
    goto LABEL_12;
LABEL_14:
  dword_4CF7F40 = v14 + v17;
  return result;
}
