// Function: sub_5D3D20
// Address: 0x5d3d20
//
__int64 __fastcall sub_5D3D20(unsigned __int8 a1, unsigned __int8 a2, int a3)
{
  __int64 result; // rax
  unsigned int v4; // ebx
  char *v5; // r14
  unsigned int v6; // r12d
  int v7; // edi

  result = dword_4F06BA0 - (a2 + (unsigned int)a1) % dword_4F06BA0;
  v4 = a3 - a2;
  if ( a3 != a2 )
  {
    do
    {
      v5 = "har";
      if ( (unsigned int)result > v4 )
        LODWORD(result) = v4;
      v6 = result;
      putc(32, stream);
      ++dword_4CF7F40;
      v7 = 99;
      do
      {
        ++v5;
        putc(v7, stream);
        v7 = *(v5 - 1);
      }
      while ( *(v5 - 1) );
      dword_4CF7F40 += 4;
      putc(58, stream);
      ++dword_4CF7F40;
      sub_5D32F0(v6);
      putc(59, stream);
      ++dword_4CF7F40;
      result = dword_4F06BA0;
      v4 -= v6;
    }
    while ( v4 );
  }
  return result;
}
