// Function: sub_6145B0
// Address: 0x6145b0
//
unsigned __int64 __fastcall sub_6145B0(char *src, unsigned __int64 *a2, _QWORD *a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rdx
  char v6; // cl
  size_t v7; // rbx
  unsigned __int64 v8; // r12
  char *v9; // rax

  result = strlen(src);
  *a3 = 0;
  *a2 = (unsigned __int64)src;
  if ( result )
  {
    v5 = result;
    v6 = 0;
    v7 = 0;
    do
    {
      while ( 1 )
      {
        result = (unsigned __int8)src[v7];
        v8 = v7 + 1;
        if ( (_BYTE)result != 92 )
          break;
        v7 += 2LL;
        if ( v5 <= v7 )
          return result;
      }
      if ( (_BYTE)result == v6 )
      {
        v6 = 0;
      }
      else if ( !v6 && ((_BYTE)result == 34 || (_BYTE)result == 39) )
      {
        v6 = src[v7];
      }
      else if ( (_BYTE)result == 61 )
      {
        v9 = (char *)sub_822B10(v7 + 1);
        result = (unsigned __int64)strncpy(v9, src, v7);
        *(_BYTE *)(result + v7) = 0;
        *a2 = result;
        *a3 = &src[v8];
        return result;
      }
      ++v7;
    }
    while ( v5 > v8 );
  }
  return result;
}
