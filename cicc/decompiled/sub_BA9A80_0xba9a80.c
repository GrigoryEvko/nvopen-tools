// Function: sub_BA9A80
// Address: 0xba9a80
//
void __fastcall sub_BA9A80(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  __int64 v4; // rdi
  _QWORD *i; // r12
  __int64 v6; // rdi
  _QWORD *j; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  _QWORD *k; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rdx

  v1 = a1 + 3;
  v2 = (_QWORD *)a1[4];
  if ( v2 != a1 + 3 )
  {
    do
    {
      v4 = (__int64)(v2 - 7);
      if ( !v2 )
        v4 = 0;
      sub_B2CA40(v4, 1);
      v2 = (_QWORD *)v2[1];
    }
    while ( v1 != v2 );
  }
  for ( i = (_QWORD *)a1[2]; a1 + 1 != i; i = (_QWORD *)i[1] )
  {
    v6 = (__int64)(i - 7);
    if ( !i )
      v6 = 0;
    sub_B30220(v6);
  }
  for ( j = (_QWORD *)a1[6]; a1 + 5 != j; j = (_QWORD *)j[1] )
  {
    if ( !j )
      BUG();
    v8 = 32LL * (*((_DWORD *)j - 11) & 0x7FFFFFF);
    if ( (*((_BYTE *)j - 41) & 0x40) != 0 )
    {
      v9 = *(j - 7);
      v10 = (_QWORD *)(v9 + v8);
    }
    else
    {
      v10 = j - 6;
      v9 = (__int64)&j[v8 / 0xFFFFFFFFFFFFFFF8LL - 6];
    }
    if ( (_QWORD *)v9 != v10 )
    {
      do
      {
        if ( *(_QWORD *)v9 )
        {
          v11 = *(_QWORD *)(v9 + 8);
          **(_QWORD **)(v9 + 16) = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 + 16);
        }
        *(_QWORD *)v9 = 0;
        v9 += 32;
      }
      while ( v10 != (_QWORD *)v9 );
    }
  }
  for ( k = (_QWORD *)a1[8]; a1 + 7 != k; k = (_QWORD *)k[1] )
  {
    if ( !k )
      BUG();
    v13 = 32LL * (*((_DWORD *)k - 13) & 0x7FFFFFF);
    if ( (*((_BYTE *)k - 49) & 0x40) != 0 )
    {
      v14 = *(k - 8);
      v15 = (_QWORD *)(v14 + v13);
    }
    else
    {
      v15 = k - 7;
      v14 = (__int64)&k[v13 / 0xFFFFFFFFFFFFFFF8LL - 7];
    }
    for ( ; (_QWORD *)v14 != v15; v14 += 32 )
    {
      if ( *(_QWORD *)v14 )
      {
        v16 = *(_QWORD *)(v14 + 8);
        **(_QWORD **)(v14 + 16) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *(_QWORD *)(v14 + 16);
      }
      *(_QWORD *)v14 = 0;
    }
  }
}
