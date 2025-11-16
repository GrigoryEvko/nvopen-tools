// Function: sub_5D6C90
// Address: 0x5d6c90
//
int __fastcall sub_5D6C90(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rax
  char *v4; // r12
  int v5; // edi
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rax
  char v16; // [rsp+Bh] [rbp-25h] BYREF
  unsigned int v17[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *(_DWORD **)(a1 + 160);
  v17[0] = 1;
  if ( *((_BYTE *)v3 + 140) != 14 )
  {
    if ( (*(_BYTE *)(a1 + 88) & 8) == 0 )
    {
      if ( !dword_4CF7EA0 )
        goto LABEL_2;
      sub_5D3EB0("#if 0");
      if ( dword_4CF7F40 )
        sub_5D37C0("#if 0", a2);
      v3 = dword_4F07508;
      dword_4CF7F3C = 0;
      dword_4CF7F44 = 0;
      qword_4CF7F48 = 0;
      dword_4F07508[0] = 0;
      LOWORD(dword_4F07508[1]) = 0;
    }
    if ( (*(_BYTE *)(a1 + 143) & 0x10) != 0 )
    {
      v3 = &dword_4F068D4;
      if ( !dword_4F068D4 )
        LODWORD(v3) = sub_5D3EB0("#include <stdarg.h>");
    }
    else if ( *(char *)(a1 + 185) >= 0 )
    {
      if ( *(char *)(a1 + 88) < 0 )
      {
        v12 = 0;
        while ( 1 )
        {
          v13 = unk_4F04C50;
          if ( unk_4F04C50 )
            v13 = qword_4CF7E98;
          v14 = sub_732D20(a1, v13, 0, v12);
          v12 = v14;
          if ( !v14 )
            break;
          sub_5D52E0(v14, v13);
        }
      }
      v4 = "ypedef ";
      sub_5D45D0((unsigned int *)(a1 + 64));
      v5 = 116;
      do
      {
        ++v4;
        putc(v5, stream);
        v5 = *(v4 - 1);
      }
      while ( *(v4 - 1) );
      dword_4CF7F40 += 8;
      byte_4CF7D77 = 1;
      if ( dword_4F077B8 )
      {
        v8 = sub_746E90(a1, &v16);
        v9 = *(_QWORD *)(a1 + 160);
        v10 = v8;
        sub_74A390(v9, 0, 1, 0, 0, &qword_4CF7CE0);
        sub_5D5A80(a1, 0);
        sub_74D110(v9, 0, 0, &qword_4CF7CE0);
        if ( v10 )
          *(_BYTE *)(v10 + 160) = v16;
      }
      else
      {
        v6 = *(_QWORD *)(a1 + 160);
        sub_74A390(v6, 0, 1, 0, 0, &qword_4CF7CE0);
        sub_5D5A80(a1, 0);
        sub_74D110(v6, 0, 0, &qword_4CF7CE0);
      }
      v11 = sub_8D21F0(*(_QWORD *)(a1 + 160));
      byte_4CF7D77 = 0;
      if ( *(_BYTE *)(v11 + 140) == 15 )
        sub_749E60(v11, v17, &qword_4CF7CE0);
      sub_74F590(a1, v17[0], &qword_4CF7CE0);
      LODWORD(v3) = putc(59, stream);
      ++dword_4CF7F40;
    }
    v7 = (unsigned int)dword_4CF7EA0;
    if ( dword_4CF7EA0 && (*(_BYTE *)(a1 + 88) & 8) == 0 )
    {
      sub_5D3EB0("#endif");
      if ( dword_4CF7F40 )
        sub_5D37C0("#endif", v7);
      v3 = dword_4F07508;
      dword_4CF7F3C = 0;
      dword_4CF7F44 = 0;
      qword_4CF7F48 = 0;
      dword_4F07508[0] = 0;
      LOWORD(dword_4F07508[1]) = 0;
    }
  }
LABEL_2:
  *(_BYTE *)(a1 + 142) |= 0x40u;
  return (int)v3;
}
