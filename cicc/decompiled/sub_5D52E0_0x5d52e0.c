// Function: sub_5D52E0
// Address: 0x5d52e0
//
void __fastcall sub_5D52E0(__int64 a1, __int64 a2)
{
  char v3; // r12
  int v4; // ebx
  __int64 v5; // rdi
  int v6; // eax
  char v7; // dl
  int v8; // edi
  const char *v9; // r14
  _BYTE *v10; // rax
  _BYTE *v11; // r13
  char *v12; // r14

  if ( !*(_BYTE *)(a1 + 9) )
  {
    v3 = byte_4CF7D72;
    v4 = dword_4CF7F38;
    if ( dword_4CF7F40 )
      sub_5D37C0(a1, a2);
    v5 = a1 + 32;
    sub_5D45D0((unsigned int *)v5);
    v6 = dword_4CF7F60;
    byte_4CF7D72 = 1;
    dword_4CF7F38 = 0;
    ++dword_4CF7F60;
    v7 = *(_BYTE *)(a1 + 8);
    if ( v7 == 26 )
    {
      if ( unk_4F04C50 )
      {
        a2 = *(unsigned __int8 *)(a1 + 57);
        v5 = *(unsigned __int8 *)(a1 + 56);
        sub_5D4B20(v5, a2);
        v6 = dword_4CF7F60 - 1;
      }
    }
    else
    {
      v8 = 35;
      v9 = "pragma ";
      if ( v7 == 21 )
      {
        v12 = "ident ";
        do
        {
          a2 = (__int64)stream;
          ++v12;
          putc(v8, stream);
          v8 = *(v12 - 1);
          ++dword_4CF7F40;
        }
        while ( (_BYTE)v8 );
        byte_4CF7D70 = 1;
        v5 = *(_QWORD *)(a1 + 56);
        sub_5D5250(v5);
        byte_4CF7D70 = 0;
        v6 = dword_4CF7F60 - 1;
      }
      else
      {
        do
        {
          a2 = (__int64)stream;
          ++v9;
          putc(v8, stream);
          v8 = *(v9 - 1);
          ++dword_4CF7F40;
        }
        while ( (_BYTE)v8 );
        v10 = *(_BYTE **)(a1 + 48);
        v5 = (unsigned int)(char)*v10;
        v11 = v10 + 1;
        if ( *v10 )
        {
          do
          {
            a2 = (__int64)stream;
            ++v11;
            putc(v5, stream);
            v5 = (unsigned int)(char)*(v11 - 1);
            ++dword_4CF7F40;
          }
          while ( (_BYTE)v5 );
        }
        v6 = dword_4CF7F60 - 1;
      }
    }
    byte_4CF7D72 = v3;
    dword_4CF7F60 = v6;
    sub_5D37C0(v5, a2);
    dword_4CF7F38 = v4;
  }
}
