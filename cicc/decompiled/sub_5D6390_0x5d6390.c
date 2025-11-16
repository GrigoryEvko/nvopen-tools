// Function: sub_5D6390
// Address: 0x5d6390
//
int __fastcall sub_5D6390(__int64 a1)
{
  __int64 v1; // r12
  int v2; // edi
  char *v3; // rbx
  int result; // eax
  __int64 v5; // rax
  unsigned int v6; // esi
  char v7; // al
  FILE *v8; // rsi
  int v9; // edi
  char *v10; // rbx
  __int64 v11; // rcx
  __int64 i; // rax
  int v13; // edx
  int v14; // edi
  char *v15; // rbx
  unsigned __int64 v16; // rdi
  __int64 v17; // rdi
  int v18; // edi
  char *v19; // rbx
  char v20; // al
  int v21; // edi
  char *v22; // rbx
  char v23; // al
  char v24; // al
  char *v25; // rbx
  int v26; // edi
  char *v27; // rbx

  v1 = a1;
  if ( qword_4CF7DA0 && qword_4CF7DA0 == unk_4F04C50 && (*(_QWORD *)(a1 + 168) & 0x100008000LL) == 0x8000 )
  {
    v11 = *(_QWORD *)(qword_4CF7DA0 + 40);
    for ( i = *(_QWORD *)(qword_4CF7D98 + 40); ; i = *(_QWORD *)(i + 112) )
    {
      if ( (*(_BYTE *)(i + 172) & 1) != 0 )
      {
        if ( dword_4CF7D8C <= 0 )
        {
          if ( dword_4CF7D88 )
            i = *(_QWORD *)(i + 112);
        }
        else
        {
          v13 = 1;
          do
          {
            ++v13;
            i = *(_QWORD *)(i + 112);
          }
          while ( dword_4CF7D8C + 1 != v13 );
        }
      }
      if ( v11 == a1 )
        break;
      v11 = *(_QWORD *)(v11 + 112);
    }
    v1 = i;
  }
  if ( (*(_BYTE *)(v1 + 172) & 1) != 0 )
  {
    v2 = 116;
    v3 = "his";
    do
    {
      ++v3;
      result = putc(v2, stream);
      v2 = *(v3 - 1);
    }
    while ( *(v3 - 1) );
    dword_4CF7F40 += 4;
    return result;
  }
  if ( (*(_BYTE *)(v1 + 175) & 0x40) != 0 )
  {
    v5 = v1;
    v6 = 0;
    do
    {
      v5 = *(_QWORD *)(v5 + 112);
      ++v6;
    }
    while ( v5 && (*(_BYTE *)(v5 + 175) & 0x40) != 0 );
    return sub_5D5A80(v1, v6);
  }
  v7 = *(_BYTE *)(v1 + 156);
  v8 = stream;
  if ( (v7 & 1) != 0 && qword_4CF7EB0 == stream && (v7 & 2) == 0 && *(char *)(v1 + 173) >= 0 )
  {
    v23 = *(_BYTE *)(v1 + 170);
    if ( (v23 & 0x10) == 0 || (v23 & 0x60) == 0 )
    {
      v24 = 95;
      v25 = "_shadow_var(";
      while ( 1 )
      {
        ++v25;
        putc(v24, v8);
        v24 = *(v25 - 1);
        ++dword_4CF7F40;
        if ( !v24 )
          break;
        v8 = stream;
      }
      sub_5D62B0(v1);
      putc(44, stream);
      ++dword_4CF7F40;
      if ( (*(_BYTE *)(v1 + 170) & 0x10) != 0 )
      {
        putc(40, stream);
        ++dword_4CF7F40;
      }
      sub_5D5580(v1, 1);
      if ( (*(_BYTE *)(v1 + 170) & 0x10) != 0 )
      {
        putc(41, stream);
        ++dword_4CF7F40;
      }
      goto LABEL_20;
    }
  }
  if ( qword_4CF7EB8 == stream )
    goto LABEL_61;
  if ( (unsigned int)sub_8D2FF0(*(_QWORD *)(v1 + 120), stream) && (*(_BYTE *)(v1 + 89) & 1) == 0 )
  {
    v9 = 95;
    v10 = "_text_var(";
    do
    {
      ++v10;
      putc(v9, stream);
      v9 = *(v10 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v9 );
    sub_5D62B0(v1);
    putc(44, stream);
    ++dword_4CF7F40;
    sub_5D5580(v1, 1);
LABEL_20:
    result = putc(41, stream);
    ++dword_4CF7F40;
    return result;
  }
  if ( qword_4CF7EB8 == stream )
  {
LABEL_61:
    if ( (*(_DWORD *)(v1 + 168) & 0x10008000) == 0x10008000
      && (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 198LL) & 0x20) != 0 )
    {
      v26 = 95;
      v27 = "_val_param(";
      do
      {
        ++v27;
        putc(v26, stream);
        v26 = *(v27 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v26 );
      sub_5D5A80(v1, 0);
      result = putc(41, stream);
      ++dword_4CF7F40;
      return result;
    }
  }
  if ( (*(_BYTE *)(v1 + 156) & 8) == 0 )
    return sub_5D62B0(v1);
  v14 = 95;
  v15 = "_cuda_local_var_";
  do
  {
    ++v15;
    putc(v14, stream);
    v14 = *(v15 - 1);
    ++dword_4CF7F40;
  }
  while ( (_BYTE)v14 );
  sub_5D32F0(*(unsigned int *)(v1 + 64));
  putc(95, stream);
  v16 = *(unsigned __int16 *)(v1 + 68);
  ++dword_4CF7F40;
  sub_5D32F0(v16);
  putc(95, stream);
  v17 = *(_QWORD *)(v1 + 120);
  ++dword_4CF7F40;
  if ( (*(_BYTE *)(v17 + 140) & 0xFB) == 8
    && (sub_8D4C10(v17, unk_4F077C4 != 2) & 1) != 0
    && ((v20 = *(_BYTE *)(v1 + 177), v20 == 3)
     || v20 == 2 && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v1 + 184) + 48LL) - 1) <= 1u) )
  {
    v21 = 99;
    v22 = "onst";
    do
    {
      ++v22;
      putc(v21, stream);
      v21 = *(v22 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v21 );
  }
  else
  {
    v18 = 110;
    v19 = "on_const";
    do
    {
      ++v19;
      putc(v18, stream);
      v18 = *(v19 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v18 );
  }
  putc(95, stream);
  ++dword_4CF7F40;
  return sub_5D5A80(v1, 0);
}
