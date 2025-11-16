// Function: sub_C475D0
// Address: 0xc475d0
//
void __fastcall sub_C475D0(_QWORD *src, unsigned int a2, unsigned int a3)
{
  unsigned int v4; // edi
  size_t v5; // r13
  char v6; // r9
  unsigned int v7; // eax
  __int64 *v8; // r8
  __int64 v9; // rdx
  unsigned int v10; // eax

  if ( a3 )
  {
    v4 = a3 >> 6;
    if ( a3 >> 6 > a2 )
      v4 = a2;
    v5 = v4;
    v6 = a3 & 0x3F;
    if ( (a3 & 0x3F) != 0 )
    {
      while ( 1 )
      {
        v10 = a2--;
        if ( v4 >= v10 )
          break;
        v7 = ~v4 + v10;
        v8 = &src[a2];
        v9 = src[v7] << v6;
        *v8 = v9;
        if ( a2 > v4 )
          *v8 = (src[v7 - 1] >> (64 - v6)) | v9;
      }
    }
    else
    {
      memmove(&src[v5], src, 8 * (a2 - v4));
    }
    memset(src, 0, v5 * 8);
  }
}
