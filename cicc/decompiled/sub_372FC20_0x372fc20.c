// Function: sub_372FC20
// Address: 0x372fc20
//
const char *__fastcall sub_372FC20(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rcx

  v1 = *(__int64 **)(a1 + 8);
  if ( v1 )
  {
    v2 = *v1;
    do
    {
      v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v3 )
        break;
      if ( *(_WORD *)(v3 + 12) == 3 )
      {
        v5 = *(_QWORD *)(v3 + 16);
        v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v5 & 4) != 0 )
          return *(const char **)(v6 + 24);
        else
          return (const char *)(v6 + 32);
      }
      v2 = *(_QWORD *)v3;
    }
    while ( (v2 & 4) == 0 );
  }
  return byte_3F871B3;
}
