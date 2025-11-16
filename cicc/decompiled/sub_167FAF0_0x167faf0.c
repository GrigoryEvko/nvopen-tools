// Function: sub_167FAF0
// Address: 0x167faf0
//
__int64 __fastcall sub_167FAF0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r12

  result = *(unsigned int *)(a1 + 16);
  if ( !(_DWORD)result )
    goto LABEL_2;
  v3 = *(_QWORD *)(a1 + 8);
  result = 3LL * *(unsigned int *)(a1 + 24);
  v4 = v3 + 24LL * *(unsigned int *)(a1 + 24);
  if ( v3 == v4 )
    goto LABEL_2;
  do
  {
    result = *(unsigned int *)(v3 + 12);
    if ( !(_DWORD)result )
    {
      if ( *(_QWORD *)v3 == -1 )
        goto LABEL_20;
      while ( 1 )
      {
LABEL_8:
        if ( v4 == v3 )
          goto LABEL_2;
        if ( *(_DWORD *)(v3 + 8) )
          result = (__int64)memcpy((char *)a2 + *(_QWORD *)(v3 + 16), *(const void **)v3, *(unsigned int *)(v3 + 8));
        v3 += 24;
        if ( v3 == v4 )
          goto LABEL_2;
        result = *(unsigned int *)(v3 + 12);
        if ( !(_DWORD)result )
          goto LABEL_17;
        while ( (_DWORD)result == 1 && *(_QWORD *)v3 == -2 )
        {
          while ( 1 )
          {
            v3 += 24;
            if ( v4 == v3 )
              goto LABEL_2;
            result = *(unsigned int *)(v3 + 12);
            if ( (_DWORD)result )
              break;
LABEL_17:
            if ( *(_QWORD *)v3 != -1 )
              goto LABEL_8;
          }
        }
      }
    }
    if ( (_DWORD)result != 1 || *(_QWORD *)v3 != -2 )
      goto LABEL_8;
LABEL_20:
    v3 += 24;
  }
  while ( v4 != v3 );
LABEL_2:
  if ( *(_DWORD *)(a1 + 40) == 1 )
  {
    result = *(_QWORD *)(a1 + 32);
    *a2 = result;
  }
  return result;
}
