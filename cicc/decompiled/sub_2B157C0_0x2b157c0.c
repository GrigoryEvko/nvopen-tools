// Function: sub_2B157C0
// Address: 0x2b157c0
//
char *__fastcall sub_2B157C0(char *a1, char *a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  char *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  char *result; // rax

  v6 = (a2 - a1) >> 5;
  v7 = (a2 - a1) >> 3;
  if ( v6 > 0 )
  {
    v8 = &a1[32 * v6];
    do
    {
      if ( **(_BYTE **)a1 == 63
        && a3 != *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1
                                       + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(*(_QWORD *)a1 + 4LL) & 0x7FFFFFF)))
                           + 8LL) )
      {
        return a1;
      }
      v9 = *((_QWORD *)a1 + 1);
      if ( *(_BYTE *)v9 == 63
        && a3 != *(_QWORD *)(*(_QWORD *)(v9 + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(v9 + 4) & 0x7FFFFFF))) + 8LL) )
      {
        return a1 + 8;
      }
      v10 = *((_QWORD *)a1 + 2);
      if ( *(_BYTE *)v10 == 63
        && a3 != *(_QWORD *)(*(_QWORD *)(v10 + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(v10 + 4) & 0x7FFFFFF))) + 8LL) )
      {
        return a1 + 16;
      }
      v11 = *((_QWORD *)a1 + 3);
      if ( *(_BYTE *)v11 == 63
        && a3 != *(_QWORD *)(*(_QWORD *)(v11 + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(v11 + 4) & 0x7FFFFFF))) + 8LL) )
      {
        return a1 + 24;
      }
      a1 += 32;
    }
    while ( a1 != v8 );
    v7 = (a2 - a1) >> 3;
  }
  if ( v7 == 2 )
  {
LABEL_24:
    if ( **(_BYTE **)a1 == 63 )
    {
      result = a1;
      if ( a3 != *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1
                                       + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(*(_QWORD *)a1 + 4LL) & 0x7FFFFFF)))
                           + 8LL) )
        return result;
    }
    a1 += 8;
    goto LABEL_26;
  }
  if ( v7 == 3 )
  {
    if ( **(_BYTE **)a1 == 63 )
    {
      result = a1;
      if ( a3 != *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1
                                       + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(*(_QWORD *)a1 + 4LL) & 0x7FFFFFF)))
                           + 8LL) )
        return result;
    }
    a1 += 8;
    goto LABEL_24;
  }
  if ( v7 != 1 )
    return a2;
LABEL_26:
  result = a2;
  if ( **(_BYTE **)a1 == 63
    && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1
                             + 32 * (a4 - (unsigned __int64)(*(_DWORD *)(*(_QWORD *)a1 + 4LL) & 0x7FFFFFF)))
                 + 8LL) != a3 )
  {
    return a1;
  }
  return result;
}
