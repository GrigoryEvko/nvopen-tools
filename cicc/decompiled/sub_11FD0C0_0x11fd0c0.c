// Function: sub_11FD0C0
// Address: 0x11fd0c0
//
unsigned __int8 *__fastcall sub_11FD0C0(unsigned __int8 *a1)
{
  __int64 v1; // r13
  unsigned __int8 *v2; // r12
  int v3; // ebx
  unsigned __int64 v4; // rbx

  v1 = 0x800000000000601LL;
  v2 = a1;
  v3 = *a1;
  if ( (_BYTE)v3 == 58 )
    return v2 + 1;
  while ( 1 )
  {
    if ( !isalnum((unsigned __int8)v3) )
    {
      v4 = (unsigned int)(v3 - 36);
      if ( (unsigned __int8)v4 > 0x3Bu || !_bittest64(&v1, v4) )
        break;
    }
    v3 = *++v2;
    if ( (_BYTE)v3 == 58 )
      return v2 + 1;
  }
  return 0;
}
