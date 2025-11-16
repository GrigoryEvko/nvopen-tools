// Function: sub_5D6FC0
// Address: 0x5d6fc0
//
__int64 sub_5D6FC0()
{
  __int64 v0; // rbx
  __int64 result; // rax
  char v2; // dl
  char v3; // r13
  int v4; // edi
  char *v5; // r12
  __int64 v6; // r12
  const char *v7; // r14
  const char *v8; // r12
  int v9; // eax
  int v10; // edi
  int v11; // r13d
  const char *v12; // r15
  int v13; // r14d
  const char *v14; // r12
  int v15; // eax
  int v16; // edi
  int v17; // r13d
  FILE *v18; // rsi

  v0 = sub_8E36B0();
  result = *(unsigned __int8 *)(v0 + 141);
  if ( (result & 4) != 0 )
    return result;
  v2 = *(_BYTE *)(v0 + 89);
  if ( (v2 & 8) == 0 )
    return result;
  if ( result & 8 | v2 & 2 || (v3 = 0, (unsigned __int8)((*(_BYTE *)(v0 + 88) & 3) - 1) <= 1u) )
  {
    v3 = 0;
    if ( *(_BYTE *)(v0 + 140) == 12 )
    {
      v3 = 1;
      sub_5D6FC0(*(_QWORD *)(v0 + 160));
    }
  }
  v4 = 116;
  v5 = "ypedef ";
  do
  {
    ++v5;
    putc(v4, stream);
    v4 = *(v5 - 1);
  }
  while ( *(v5 - 1) );
  dword_4CF7F40 += 8;
  v6 = sub_8D21C0(v0);
  if ( !sub_5D2DA0(v6) )
  {
    if ( !v3 )
    {
      sub_5D5580(v0, 0);
      goto LABEL_15;
    }
    v6 = *(_QWORD *)(v0 + 160);
    if ( (*(_BYTE *)(v6 + 89) & 8) != 0 )
    {
      v7 = *(const char **)(v6 + 8);
      v8 = v7 + 1;
      v9 = strlen(v7);
      v10 = *v7;
      v11 = v9;
      if ( *v7 )
      {
        do
        {
          ++v8;
          putc(v10, stream);
          v10 = *(v8 - 1);
        }
        while ( *(v8 - 1) );
      }
      dword_4CF7F40 += v11;
      goto LABEL_15;
    }
  }
  sub_74A390(v6, 0, 0, 0, 0, &qword_4CF7CE0);
  sub_74D110(v6, 0, 0, &qword_4CF7CE0);
LABEL_15:
  putc(32, stream);
  v12 = *(const char **)(v0 + 8);
  v13 = dword_4CF7F40 + 1;
  v14 = v12 + 1;
  ++dword_4CF7F40;
  v15 = strlen(v12);
  v16 = *v12;
  v17 = v15;
  if ( *v12 )
  {
    do
    {
      ++v14;
      putc(v16, stream);
      v16 = *(v14 - 1);
    }
    while ( *(v14 - 1) );
    v13 = dword_4CF7F40;
  }
  v18 = stream;
  dword_4CF7F40 = v13 + v17;
  putc(59, stream);
  ++dword_4CF7F40;
  result = sub_5D37C0(59, v18);
  *(_BYTE *)(v0 + 141) |= 4u;
  return result;
}
