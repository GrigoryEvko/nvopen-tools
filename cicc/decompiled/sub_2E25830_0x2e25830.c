// Function: sub_2E25830
// Address: 0x2e25830
//
bool __fastcall sub_2E25830(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 *v8; // rdi
  __int64 *v9; // rax
  unsigned int v10; // ecx
  unsigned int v11; // esi
  unsigned int v12; // edx
  __int64 v13; // rax

  v8 = (__int64 *)*a1;
  if ( a1 == v8 )
    goto LABEL_12;
  v9 = (__int64 *)a1[3];
  v10 = *(_DWORD *)(a2 + 24);
  if ( a1 == v9 )
  {
    v9 = (__int64 *)a1[1];
    v12 = v10 >> 7;
    a1[3] = (__int64)v9;
    v11 = *((_DWORD *)v9 + 4);
    if ( v10 >> 7 == v11 )
    {
      if ( a1 == v9 )
        goto LABEL_12;
      goto LABEL_17;
    }
  }
  else
  {
    v11 = *((_DWORD *)v9 + 4);
    v12 = v10 >> 7;
    if ( v10 >> 7 == v11 )
    {
LABEL_17:
      if ( (v9[((v10 >> 6) & 1) + 3] & (1LL << v10)) != 0 )
        return 1;
      goto LABEL_12;
    }
  }
  if ( v11 > v12 )
  {
    if ( v8 != v9 )
    {
      while ( 1 )
      {
        v9 = (__int64 *)v9[1];
        if ( v8 == v9 )
          break;
        if ( *((_DWORD *)v9 + 4) <= v12 )
          goto LABEL_10;
      }
    }
    a1[3] = (__int64)v9;
  }
  else
  {
    if ( a1 == v9 )
    {
LABEL_24:
      a1[3] = (__int64)v9;
      goto LABEL_12;
    }
    while ( v11 < v12 )
    {
      v9 = (__int64 *)*v9;
      if ( a1 == v9 )
        goto LABEL_24;
      v11 = *((_DWORD *)v9 + 4);
    }
LABEL_10:
    a1[3] = (__int64)v9;
    if ( a1 == v9 )
      goto LABEL_12;
  }
  if ( *((_DWORD *)v9 + 4) == v12 )
    goto LABEL_17;
LABEL_12:
  v13 = sub_2EBEE10(a4, a3);
  return (!v13 || a2 != *(_QWORD *)(v13 + 24)) && sub_2E24F20((__int64)a1, a2) != 0;
}
