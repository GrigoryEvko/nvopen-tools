// Function: sub_1648080
// Address: 0x1648080
//
void __fastcall sub_1648080(__int64 a1, _QWORD *a2, char a3)
{
  _QWORD *i; // rbx
  _QWORD *v4; // rbx
  _QWORD *j; // r13
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 *v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rsi
  _QWORD *m; // r13
  __int64 *v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // rsi
  unsigned __int64 v17; // r15
  _BYTE *v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // r15
  _QWORD *v21; // r13
  unsigned int v22; // ebx
  int v23; // r14d
  unsigned int v24; // esi
  __int64 v25; // rax
  _QWORD *v27; // [rsp+20h] [rbp-A0h]
  _QWORD *k; // [rsp+38h] [rbp-88h]
  _BYTE *v29; // [rsp+40h] [rbp-80h] BYREF
  __int64 v30; // [rsp+48h] [rbp-78h]
  _BYTE v31[112]; // [rsp+50h] [rbp-70h] BYREF

  *(_BYTE *)(a1 + 120) = a3;
  for ( i = (_QWORD *)a2[2]; i != a2 + 1; i = (_QWORD *)i[1] )
  {
    if ( !i )
      BUG();
    sub_1647410(a1, *(i - 7));
    if ( !sub_15E4F60((__int64)(i - 7)) )
      sub_1647DB0(a1, *(i - 10));
  }
  v4 = (_QWORD *)a2[6];
  for ( j = a2 + 5; j != v4; v4 = (_QWORD *)v4[1] )
  {
    if ( !v4 )
      BUG();
    sub_1647410(a1, *(v4 - 6));
    v6 = *(v4 - 9);
    if ( v6 )
      sub_1647DB0(a1, v6);
  }
  v29 = v31;
  v30 = 0x400000000LL;
  v27 = (_QWORD *)a2[4];
  if ( a2 + 3 == v27 )
  {
    v20 = a2[10];
    v21 = a2 + 9;
    if ( a2 + 9 == (_QWORD *)v20 )
      return;
  }
  else
  {
    do
    {
      if ( !v27 )
        BUG();
      v7 = *(v27 - 7);
      sub_1647410(a1, v7);
      if ( (*((_BYTE *)v27 - 33) & 0x40) != 0 )
      {
        v8 = (__int64 *)*(v27 - 8);
        v9 = &v8[3 * (*((_DWORD *)v27 - 9) & 0xFFFFFFF)];
      }
      else
      {
        v9 = v27 - 7;
        v8 = &v27[-3 * (*((_DWORD *)v27 - 9) & 0xFFFFFFF) - 7];
      }
      while ( v9 != v8 )
      {
        v7 = *v8;
        v8 += 3;
        sub_1647DB0(a1, v7);
      }
      if ( (*((_BYTE *)v27 - 38) & 1) != 0 )
      {
        sub_15E08E0((__int64)(v27 - 7), v7);
        v10 = v27[4];
        v11 = v10 + 40LL * v27[5];
        if ( (*((_BYTE *)v27 - 38) & 1) != 0 )
        {
          sub_15E08E0((__int64)(v27 - 7), v7);
          v10 = v27[4];
        }
      }
      else
      {
        v10 = v27[4];
        v11 = v10 + 40LL * v27[5];
      }
      while ( v11 != v10 )
      {
        v12 = v10;
        v10 += 40;
        sub_1647DB0(a1, v12);
      }
      for ( k = (_QWORD *)v27[3]; v27 + 2 != k; k = (_QWORD *)k[1] )
      {
        if ( !k )
          BUG();
        for ( m = (_QWORD *)k[3]; k + 2 != m; m = (_QWORD *)m[1] )
        {
          if ( !m )
            BUG();
          sub_1647410(a1, *(m - 3));
          if ( (*((_BYTE *)m - 1) & 0x40) != 0 )
          {
            v14 = (__int64 *)*(m - 4);
            v15 = &v14[3 * (*((_DWORD *)m - 1) & 0xFFFFFFF)];
          }
          else
          {
            v15 = m - 3;
            v14 = &m[-3 * (*((_DWORD *)m - 1) & 0xFFFFFFF) - 3];
          }
          while ( v15 != v14 )
          {
            while ( 1 )
            {
              v16 = *v14;
              if ( *v14 )
              {
                if ( *(_BYTE *)(v16 + 16) <= 0x17u )
                  break;
              }
              v14 += 3;
              if ( v15 == v14 )
                goto LABEL_33;
            }
            v14 += 3;
            sub_1647DB0(a1, v16);
          }
LABEL_33:
          if ( *((__int16 *)m - 3) < 0 )
            sub_161F980((__int64)(m - 3), (__int64)&v29);
          v17 = (unsigned __int64)v29;
          v18 = &v29[16 * (unsigned int)v30];
          if ( v29 != v18 )
          {
            do
            {
              v19 = *(_QWORD *)(v17 + 8);
              v17 += 16LL;
              sub_1647B40(a1, v19);
            }
            while ( v18 != (_BYTE *)v17 );
          }
          LODWORD(v30) = 0;
        }
      }
      v27 = (_QWORD *)v27[1];
    }
    while ( a2 + 3 != v27 );
    v20 = a2[10];
    v21 = a2 + 9;
    if ( a2 + 9 == (_QWORD *)v20 )
      goto LABEL_44;
  }
  do
  {
    v22 = 0;
    v23 = sub_161F520(v20);
    if ( v23 )
    {
      do
      {
        v24 = v22++;
        v25 = sub_161F530(v20, v24);
        sub_1647B40(a1, v25);
      }
      while ( v23 != v22 );
    }
    v20 = *(_QWORD *)(v20 + 8);
  }
  while ( v21 != (_QWORD *)v20 );
LABEL_44:
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
}
