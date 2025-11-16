// Function: sub_25DC970
// Address: 0x25dc970
//
bool __fastcall sub_25DC970(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  unsigned __int8 *v3; // rdi
  int v4; // eax
  unsigned __int64 v5; // rax
  bool result; // al

  v1 = 0x8000000000041LL;
  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v3 = *(unsigned __int8 **)(v2 + 24);
    v4 = *v3;
    if ( (unsigned __int8)v4 > 0x1Cu )
    {
      v5 = (unsigned int)(v4 - 34);
      if ( (unsigned __int8)v5 <= 0x33u )
      {
        if ( _bittest64(&v1, v5) )
        {
          result = sub_B49200((__int64)v3);
          if ( result )
            break;
        }
      }
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  return result;
}
