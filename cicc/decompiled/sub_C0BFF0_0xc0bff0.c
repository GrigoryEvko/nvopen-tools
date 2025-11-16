// Function: sub_C0BFF0
// Address: 0xc0bff0
//
__int64 __fastcall sub_C0BFF0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  int v5; // eax
  int v6; // eax

  if ( !*(_DWORD *)(a1 + 16) )
    goto LABEL_2;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 + 24LL * *(unsigned int *)(a1 + 24);
  if ( v3 == v4 )
    goto LABEL_2;
  do
  {
    v5 = *(_DWORD *)(v3 + 12);
    if ( !v5 )
    {
      if ( *(_QWORD *)v3 == -1 )
        goto LABEL_21;
      while ( 1 )
      {
LABEL_9:
        if ( v4 == v3 )
          goto LABEL_2;
        if ( *(_DWORD *)(v3 + 8) )
          memcpy((char *)a2 + *(_QWORD *)(v3 + 16), *(const void **)v3, *(unsigned int *)(v3 + 8));
        v3 += 24;
        if ( v3 == v4 )
          goto LABEL_2;
        v6 = *(_DWORD *)(v3 + 12);
        if ( !v6 )
          goto LABEL_18;
        while ( v6 == 1 && *(_QWORD *)v3 == -2 )
        {
          while ( 1 )
          {
            v3 += 24;
            if ( v4 == v3 )
              goto LABEL_2;
            v6 = *(_DWORD *)(v3 + 12);
            if ( v6 )
              break;
LABEL_18:
            if ( *(_QWORD *)v3 != -1 )
              goto LABEL_9;
          }
        }
      }
    }
    if ( v5 != 1 || *(_QWORD *)v3 != -2 )
      goto LABEL_9;
LABEL_21:
    v3 += 24;
  }
  while ( v4 != v3 );
LABEL_2:
  result = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)result == 1 )
  {
    result = *(_QWORD *)(a1 + 32);
    *a2 = result;
  }
  else if ( (_DWORD)result == 8 )
  {
    result = _byteswap_ulong(*(_QWORD *)(a1 + 32));
    *a2 = result;
  }
  return result;
}
