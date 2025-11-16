// Function: sub_5D3F60
// Address: 0x5d3f60
//
void __fastcall sub_5D3F60(__int64 a1)
{
  int v1; // r12d
  char *v2; // rax
  int v3; // edi
  char *v4; // rbx

  switch ( (char)a1 )
  {
    case 0:
      return;
    case 1:
      v1 = 6;
      v3 = 101;
      v4 = "xtern";
      goto LABEL_4;
    case 2:
      v1 = 6;
      v2 = "static";
      goto LABEL_3;
    case 3:
      v1 = 0;
      goto LABEL_5;
    case 4:
      v1 = 7;
      v2 = "typedef";
      goto LABEL_3;
    case 5:
      v1 = 8;
      v2 = "register";
LABEL_3:
      v3 = *v2;
      v4 = v2 + 1;
      do
      {
LABEL_4:
        ++v4;
        putc(v3, stream);
        v3 = *(v4 - 1);
      }
      while ( *(v4 - 1) );
LABEL_5:
      dword_4CF7F40 += v1;
      putc(32, stream);
      ++dword_4CF7F40;
      return;
    default:
      sub_721090(a1);
  }
}
