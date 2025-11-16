// Function: sub_13E7650
// Address: 0x13e7650
//
__int64 __fastcall sub_13E7650(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rdx
  _QWORD *v6; // rax
  signed __int64 v7; // rdx
  _QWORD *v8; // rdx

  v5 = 24LL * (*((_DWORD *)a1 + 5) & 0xFFFFFFF);
  v6 = &a1[v5 / 0xFFFFFFFFFFFFFFF8LL];
  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
  {
    v6 = (_QWORD *)*(a1 - 1);
    a1 = &v6[(unsigned __int64)v5 / 8];
  }
  v7 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
  if ( v7 >> 2 )
  {
    v8 = &v6[12 * (v7 >> 2)];
    while ( a2 != *v6 )
    {
      if ( a2 == v6[3] )
      {
        LOBYTE(a5) = a1 != v6 + 3;
        return a5;
      }
      if ( a2 == v6[6] )
      {
        LOBYTE(a5) = a1 != v6 + 6;
        return a5;
      }
      if ( a2 == v6[9] )
      {
        LOBYTE(a5) = a1 != v6 + 9;
        return a5;
      }
      v6 += 12;
      if ( v8 == v6 )
      {
        v7 = 0xAAAAAAAAAAAAAAABLL * (a1 - v6);
        goto LABEL_13;
      }
    }
    goto LABEL_10;
  }
LABEL_13:
  if ( v7 != 2 )
  {
    if ( v7 != 3 )
    {
      a5 = 0;
      if ( v7 != 1 )
        return a5;
      goto LABEL_16;
    }
    if ( a2 == *v6 )
      goto LABEL_10;
    v6 += 3;
  }
  if ( a2 == *v6 )
    goto LABEL_10;
  v6 += 3;
LABEL_16:
  a5 = 0;
  if ( a2 == *v6 )
  {
LABEL_10:
    LOBYTE(a5) = a1 != v6;
    return a5;
  }
  return 0;
}
