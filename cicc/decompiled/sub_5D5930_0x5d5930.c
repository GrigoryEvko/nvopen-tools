// Function: sub_5D5930
// Address: 0x5d5930
//
__int64 __fastcall sub_5D5930(__int64 a1)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  int v4; // edi
  char *v5; // rbx
  char v6; // al
  char *v7; // rbx
  int v8; // edi
  char *v9; // rbx
  FILE *v10; // rsi
  int v11; // edi
  char *v12; // rbx

  if ( !a1 || *(_BYTE *)(a1 + 28) != 3 )
    return 0;
  v2 = *(_QWORD *)(a1 + 32);
  v3 = sub_5D5930(*(_QWORD *)(v2 + 40)) + 1;
  if ( (*(_BYTE *)(v2 + 124) & 2) != 0 )
  {
    v11 = 105;
    v12 = "nline ";
    do
    {
      ++v12;
      putc(v11, stream);
      v11 = *(v12 - 1);
    }
    while ( *(v12 - 1) );
    dword_4CF7F40 += 7;
  }
  v4 = 110;
  v5 = "amespace ";
  do
  {
    ++v5;
    putc(v4, stream);
    v4 = *(v5 - 1);
    ++dword_4CF7F40;
  }
  while ( (_BYTE)v4 );
  v6 = *(_BYTE *)(v2 + 89);
  if ( (v6 & 0x40) != 0 || ((v6 & 8) != 0 ? (v7 = *(char **)(v2 + 24)) : (v7 = *(char **)(v2 + 8)), !v7) )
  {
    v8 = 95;
    v9 = "NV_ANON_NAMESPACE";
    do
    {
LABEL_14:
      ++v9;
      putc(v8, stream);
      v8 = *(v9 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v8 );
    goto LABEL_12;
  }
  v8 = *v7;
  v9 = v7 + 1;
  if ( (_BYTE)v8 )
    goto LABEL_14;
LABEL_12:
  v10 = stream;
  putc(123, stream);
  ++dword_4CF7F40;
  sub_5D37C0(123, v10);
  return v3;
}
