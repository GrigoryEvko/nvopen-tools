// Function: sub_5D4B20
// Address: 0x5d4b20
//
__int64 __fastcall sub_5D4B20(__int64 a1, __int64 a2)
{
  int v2; // r13d
  char v3; // r12
  __int64 v4; // rdi
  char *v5; // r14
  char *v6; // r12
  __int64 v7; // rdi
  char *v8; // rbx
  FILE *v9; // rsi
  __int64 result; // rax
  char *v11; // rbx
  char *v12; // r12
  char *v13; // rbx
  char *v14; // r12

  v2 = dword_4CF7F38;
  v3 = a1;
  if ( dword_4CF7F40 )
    sub_5D37C0(a1, a2);
  ++dword_4CF7F60;
  LODWORD(v4) = 35;
  v5 = "pragma STDC ";
  dword_4CF7F38 = 0;
  do
  {
    ++v5;
    putc(v4, stream);
    v4 = (unsigned int)*(v5 - 1);
    ++dword_4CF7F40;
  }
  while ( (_BYTE)v4 );
  switch ( v3 )
  {
    case 2:
      LODWORD(v4) = 70;
      v14 = "ENV_ACCESS ";
      do
      {
        ++v14;
        putc(v4, stream);
        v4 = (unsigned int)*(v14 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v4 );
      if ( !unk_4F04C50 )
        byte_4CF7CD1 = a2;
      break;
    case 3:
      LODWORD(v4) = 67;
      v6 = "X_LIMITED_RANGE ";
      do
      {
        ++v6;
        putc(v4, stream);
        v4 = (unsigned int)*(v6 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v4 );
      if ( !unk_4F04C50 )
        byte_4CF7CD0 = a2;
      break;
    case 1:
      LODWORD(v4) = 70;
      v12 = "P_CONTRACT ";
      do
      {
        ++v12;
        putc(v4, stream);
        v4 = (unsigned int)*(v12 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v4 );
      if ( !unk_4F04C50 )
        byte_4CF7CD2 = a2;
      break;
    default:
      goto LABEL_8;
  }
  switch ( (_BYTE)a2 )
  {
    case 2:
      LODWORD(v7) = 79;
      v13 = "N";
      do
      {
        v9 = stream;
        ++v13;
        putc(v7, stream);
        v7 = (unsigned int)*(v13 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v7 );
      break;
    case 3:
      LODWORD(v7) = 68;
      v8 = "EFAULT";
      do
      {
        v9 = stream;
        ++v8;
        putc(v7, stream);
        v7 = (unsigned int)*(v8 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v7 );
      break;
    case 1:
      LODWORD(v7) = 79;
      v11 = "FF";
      do
      {
        v9 = stream;
        ++v11;
        putc(v7, stream);
        v7 = (unsigned int)*(v11 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v7 );
      break;
    default:
LABEL_8:
      sub_721090(v4);
  }
  --dword_4CF7F60;
  result = sub_5D37C0(v7, v9);
  dword_4CF7F38 = v2;
  return result;
}
