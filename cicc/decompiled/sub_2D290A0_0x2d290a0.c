// Function: sub_2D290A0
// Address: 0x2d290a0
//
__int64 *__fastcall sub_2D290A0(__int64 a1)
{
  __int64 *result; // rax
  __int64 *v2; // rcx
  __int64 v3; // rdx

  result = *(__int64 **)(a1 + 16);
  v2 = *(__int64 **)(a1 + 24);
  if ( result != v2 )
  {
    v3 = *result;
    if ( *result == -4096 )
      goto LABEL_8;
LABEL_3:
    if ( v3 == -8192 && result[1] == -8192 )
    {
      do
      {
        result += 44;
        *(_QWORD *)(a1 + 16) = result;
        if ( result == v2 )
          break;
        v3 = *result;
        if ( *result != -4096 )
          goto LABEL_3;
LABEL_8:
        ;
      }
      while ( result[1] == -4096 );
    }
  }
  return result;
}
