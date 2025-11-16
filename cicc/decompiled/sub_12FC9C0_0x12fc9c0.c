// Function: sub_12FC9C0
// Address: 0x12fc9c0
//
size_t __fastcall sub_12FC9C0(unsigned __int8 *a1, __int64 a2, const char *a3)
{
  unsigned __int8 *v4; // rbx
  size_t result; // rax
  size_t v6; // r14
  int v7; // r12d

  v4 = a1;
  result = strlen(a3);
  if ( a2 )
  {
    v6 = result;
    do
    {
      v7 = *v4;
      switch ( (char)v7 )
      {
        case 'J':
        case 'a':
        case 'b':
        case 'd':
        case 'e':
        case 'g':
        case 'h':
        case 'l':
        case 'm':
        case 'x':
          result = (size_t)strchr(a3, (char)v7);
          if ( !result )
          {
            a3[v6] = v7;
            a3[++v6] = 0;
          }
          break;
        default:
          result = (unsigned int)(v7 - 74);
          break;
      }
      ++v4;
    }
    while ( &a1[a2] != v4 );
  }
  return result;
}
