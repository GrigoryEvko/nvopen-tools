// Function: sub_5D4940
// Address: 0x5d4940
//
__int64 __fastcall sub_5D4940(__int64 a1, FILE *a2)
{
  int IO_read_end; // r9d
  FILE *v4; // r12
  int v5; // edi
  char *v6; // rbx
  FILE *v7; // rdi
  __int64 result; // rax
  FILE *v9; // rdi
  FILE *v10; // rax
  FILE *v11; // rax

  IO_read_end = (int)a2->_IO_read_end;
  LODWORD(a2->_IO_read_ptr) = 0;
  v4 = stream;
  if ( IO_read_end )
  {
    a2 = stream;
    if ( !dword_4CF7CD4 )
    {
      v9 = qword_4CF7EA8;
      if ( !qword_4CF7EA8 )
      {
        v11 = (FILE *)sub_721330(0, stream);
        qword_4CF7F00 = 0;
        qword_4CF7EA8 = v11;
        v9 = v11;
        qword_4CF7F08 = 0;
        dword_4CF7F10 = 0;
      }
      sub_5D3B20(v9);
      a2 = stream;
    }
    v5 = 125;
    v6 = "}";
    while ( 1 )
    {
      putc(v5, a2);
      v5 = *v6++;
      if ( !(_BYTE)v5 )
        break;
      a2 = stream;
    }
    dword_4CF7F40 += 2;
    if ( stream != v4 )
    {
      sub_5D3B20(v4);
      v4 = stream;
    }
  }
  if ( dword_4CF7CD4 )
  {
    result = (unsigned int)dword_4CF7EA0;
    if ( !dword_4CF7EA0 )
      return result;
  }
  else
  {
    v7 = qword_4CF7EA8;
    if ( !qword_4CF7EA8 )
    {
      v10 = (FILE *)sub_721330(0, a2);
      qword_4CF7F00 = 0;
      qword_4CF7EA8 = v10;
      v7 = v10;
      qword_4CF7F08 = 0;
      dword_4CF7F10 = 0;
    }
    result = sub_5D3B20(v7);
    a2 = (FILE *)(unsigned int)dword_4CF7EA0;
    if ( !dword_4CF7EA0 )
      goto LABEL_12;
  }
  if ( (*(_BYTE *)(a1 + 88) & 8) == 0 )
  {
    sub_5D3EB0("#endif");
    if ( dword_4CF7F40 )
      sub_5D37C0("#endif", a2);
    result = (__int64)dword_4F07508;
    dword_4CF7F3C = 0;
    dword_4CF7F44 = 0;
    qword_4CF7F48 = 0;
    dword_4F07508[0] = 0;
    LOWORD(dword_4F07508[1]) = 0;
    if ( stream != v4 )
      return sub_5D3B20(v4);
    return result;
  }
LABEL_12:
  if ( stream != v4 )
    return sub_5D3B20(v4);
  return result;
}
