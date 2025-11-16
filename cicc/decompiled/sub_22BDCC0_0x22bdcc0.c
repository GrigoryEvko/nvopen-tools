// Function: sub_22BDCC0
// Address: 0x22bdcc0
//
__int64 __fastcall sub_22BDCC0(char *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rdx
  char *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  char *v9; // rdx

  v5 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  v6 = &a1[-v5];
  if ( (a1[7] & 0x40) != 0 )
  {
    v6 = (char *)*((_QWORD *)a1 - 1);
    a1 = &v6[v5];
  }
  v7 = v5 >> 5;
  v8 = v5 >> 7;
  if ( v8 )
  {
    v9 = &v6[128 * v8];
    while ( a2 != *(_QWORD *)v6 )
    {
      if ( a2 == *((_QWORD *)v6 + 4) )
      {
        v6 += 32;
        goto LABEL_10;
      }
      if ( a2 == *((_QWORD *)v6 + 8) )
      {
        LOBYTE(a5) = a1 != v6 + 64;
        return a5;
      }
      if ( a2 == *((_QWORD *)v6 + 12) )
      {
        LOBYTE(a5) = a1 != v6 + 96;
        return a5;
      }
      v6 += 128;
      if ( v9 == v6 )
      {
        v7 = (a1 - v6) >> 5;
        goto LABEL_13;
      }
    }
    goto LABEL_10;
  }
LABEL_13:
  if ( v7 == 2 )
    goto LABEL_20;
  if ( v7 != 3 )
  {
    a5 = 0;
    if ( v7 != 1 )
      return a5;
    goto LABEL_16;
  }
  if ( a2 != *(_QWORD *)v6 )
  {
    v6 += 32;
LABEL_20:
    if ( a2 == *(_QWORD *)v6 )
      goto LABEL_17;
    v6 += 32;
LABEL_16:
    a5 = 0;
    if ( a2 != *(_QWORD *)v6 )
      return a5;
LABEL_17:
    LOBYTE(a5) = v6 != a1;
    return a5;
  }
LABEL_10:
  LOBYTE(a5) = a1 != v6;
  return a5;
}
