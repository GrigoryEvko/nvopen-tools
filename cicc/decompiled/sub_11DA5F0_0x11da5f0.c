// Function: sub_11DA5F0
// Address: 0x11da5f0
//
__int64 __fastcall sub_11DA5F0(char *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rdx
  char *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  char *v9; // rdx

  v5 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  if ( (a1[7] & 0x40) != 0 )
  {
    v6 = (char *)*((_QWORD *)a1 - 1);
    a1 = &v6[v5];
  }
  else
  {
    v6 = &a1[-v5];
  }
  v7 = v5 >> 5;
  v8 = v5 >> 7;
  if ( v8 )
  {
    v9 = &v6[128 * v8];
    while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL) != 5 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v6 + 4) + 8LL) + 8LL) == 5 )
      {
        LOBYTE(a5) = a1 != v6 + 32;
        return a5;
      }
      if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v6 + 8) + 8LL) + 8LL) == 5 )
      {
        LOBYTE(a5) = a1 != v6 + 64;
        return a5;
      }
      if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v6 + 12) + 8LL) + 8LL) == 5 )
      {
        LOBYTE(a5) = a1 != v6 + 96;
        return a5;
      }
      v6 += 128;
      if ( v9 == v6 )
      {
        v7 = (a1 - v6) >> 5;
        goto LABEL_14;
      }
    }
    goto LABEL_10;
  }
LABEL_14:
  if ( v7 == 2 )
  {
LABEL_21:
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL) != 5 )
    {
      v6 += 32;
LABEL_17:
      a5 = 0;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL) != 5 )
        return a5;
      goto LABEL_10;
    }
    goto LABEL_10;
  }
  if ( v7 != 3 )
  {
    a5 = 0;
    if ( v7 != 1 )
      return a5;
    goto LABEL_17;
  }
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL) != 5 )
  {
    v6 += 32;
    goto LABEL_21;
  }
LABEL_10:
  LOBYTE(a5) = v6 != a1;
  return a5;
}
