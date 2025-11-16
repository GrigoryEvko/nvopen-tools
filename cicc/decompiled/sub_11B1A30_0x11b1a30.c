// Function: sub_11B1A30
// Address: 0x11b1a30
//
_QWORD *__fastcall sub_11B1A30(_QWORD *a1, __int64 *a2)
{
  char v4; // dl
  _QWORD *v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rax
  _QWORD *v8; // rsi
  bool v9; // al
  bool v10; // di
  __int64 *v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx

  v4 = a2[1] & 1;
  if ( *((_DWORD *)a2 + 2) >> 1 )
  {
    if ( v4 )
    {
      v5 = a2 + 2;
      v6 = 8;
    }
    else
    {
      v5 = (_QWORD *)a2[2];
      v6 = 2LL * *((unsigned int *)a2 + 6);
    }
    *a1 = a2;
    v7 = *a2;
    v8 = &v5[v6];
    a1[2] = v5;
    a1[1] = v7;
    a1[3] = v8;
    if ( v8 == v5 )
      return a1;
    v9 = 0;
    while ( 1 )
    {
      v10 = v9;
      v9 = *v5 == -8192 || *v5 == -4096;
      if ( !v9 )
        break;
      v5 += 2;
      if ( v8 == v5 )
      {
        a1[2] = v8;
        return a1;
      }
    }
    if ( !v10 )
    {
      return a1;
    }
    else
    {
      a1[2] = v5;
      return a1;
    }
  }
  else
  {
    if ( v4 )
    {
      v12 = a2 + 2;
      v13 = 8;
    }
    else
    {
      v12 = (__int64 *)a2[2];
      v13 = 2LL * *((unsigned int *)a2 + 6);
    }
    *a1 = a2;
    v14 = *a2;
    v15 = &v12[v13];
    a1[2] = v15;
    a1[1] = v14;
    a1[3] = v15;
    return a1;
  }
}
