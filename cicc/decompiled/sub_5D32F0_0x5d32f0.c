// Function: sub_5D32F0
// Address: 0x5d32f0
//
int __fastcall sub_5D32F0(unsigned __int64 a1)
{
  unsigned __int64 v1; // r12
  int result; // eax
  int v3; // edi
  int v4; // r12d
  char *v5; // rbx
  char s; // [rsp+0h] [rbp-60h] BYREF
  char v7; // [rsp+1h] [rbp-5Fh] BYREF

  v1 = a1;
  if ( a1 <= 9 )
    goto LABEL_10;
  if ( a1 <= 0x63 )
  {
LABEL_9:
    putc((char)(v1 / 0xA + 48), stream);
    ++dword_4CF7F40;
    LOBYTE(v1) = v1 % 0xA;
LABEL_10:
    result = putc((char)(v1 + 48), stream);
    ++dword_4CF7F40;
    return result;
  }
  if ( a1 <= 0x3E7 )
  {
LABEL_8:
    putc((char)(v1 / 0x64 + 48), stream);
    ++dword_4CF7F40;
    v1 -= 100 * (unsigned int)(v1 / 0x64);
    goto LABEL_9;
  }
  if ( a1 <= 0x270F )
  {
LABEL_7:
    putc((char)(v1 / 0x3E8 + 48), stream);
    ++dword_4CF7F40;
    v1 -= 1000 * (unsigned int)(v1 / 0x3E8);
    goto LABEL_8;
  }
  if ( a1 <= 0x1869F )
  {
    putc((unsigned __int8)(a1 / 0x2710 + 48), stream);
    ++dword_4CF7F40;
    v1 = a1 - 10000 * (unsigned int)(a1 / 0x2710);
    goto LABEL_7;
  }
  sub_622470(a1, &s);
  result = strlen(&s);
  v3 = s;
  v4 = result;
  if ( s )
  {
    v5 = &v7;
    do
    {
      ++v5;
      result = putc(v3, stream);
      v3 = *(v5 - 1);
    }
    while ( *(v5 - 1) );
  }
  dword_4CF7F40 += v4;
  return result;
}
