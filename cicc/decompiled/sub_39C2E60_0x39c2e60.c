// Function: sub_39C2E60
// Address: 0x39c2e60
//
const char *__fastcall sub_39C2E60(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax

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
        return (const char *)(*(_QWORD *)(v3 + 16) + 24LL);
      v2 = *(_QWORD *)v3;
    }
    while ( (v2 & 4) == 0 );
  }
  return byte_3F871B3;
}
