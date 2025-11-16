// Function: sub_16051F0
// Address: 0x16051f0
//
__int64 __fastcall sub_16051F0(__int64 a1)
{
  __int64 *v2; // rdx
  __int64 *v3; // r12
  __int64 result; // rax
  char v5; // cl
  __int64 *i; // rbx
  __int64 v7; // rdi

  do
  {
    v2 = *(__int64 **)(a1 + 1560);
    v3 = &v2[*(unsigned int *)(a1 + 1576)];
    result = *(unsigned int *)(a1 + 1568);
    if ( (_DWORD)result )
    {
      for ( ; v2 != v3; ++v2 )
      {
        result = *v2;
        if ( *v2 != -16 && result != -8 )
          break;
      }
    }
    else
    {
      v2 += *(unsigned int *)(a1 + 1576);
    }
    v5 = 0;
    while ( v2 != v3 )
    {
      while ( 1 )
      {
        for ( i = v2 + 1; i != v3; ++i )
        {
          result = *i;
          if ( *i != -8 && result != -16 )
            break;
        }
        v7 = *v2;
        v2 = i;
        if ( *(_QWORD *)(v7 + 8) )
          break;
        result = sub_159D850(v7);
        v2 = i;
        v5 = 1;
        if ( i == v3 )
          goto LABEL_11;
      }
    }
LABEL_11:
    ;
  }
  while ( v5 );
  return result;
}
