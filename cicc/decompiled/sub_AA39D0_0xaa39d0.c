// Function: sub_AA39D0
// Address: 0xaa39d0
//
char *__fastcall sub_AA39D0(__int64 a1, char *a2, char *a3)
{
  char *v4; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  _QWORD *v8; // rbx
  __int64 *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // rax
  __int64 *v14; // rdi
  size_t v15; // rdx
  char *v16; // r15
  char *v17; // rbx
  __int64 v18; // rdi

  if ( a3 == a2 )
    return a2;
  v4 = *(char **)(a1 + 8);
  v6 = v4 - a3;
  if ( a3 == v4 )
    goto LABEL_18;
  v7 = 0x6DB6DB6DB6DB6DB7LL * (v6 >> 3);
  if ( v6 <= 0 )
    goto LABEL_18;
  v8 = a3 + 16;
  v9 = (__int64 *)(a2 + 16);
  do
  {
    v13 = (_QWORD *)*(v8 - 2);
    v14 = (__int64 *)*(v9 - 2);
    if ( v13 == v8 )
    {
      v15 = *(v8 - 1);
      if ( v15 )
      {
        if ( v15 == 1 )
          *(_BYTE *)v14 = *(_BYTE *)v8;
        else
          memcpy(v14, v8, v15);
        v15 = *(v8 - 1);
        v14 = (__int64 *)*(v9 - 2);
      }
      *(v9 - 1) = v15;
      *((_BYTE *)v14 + v15) = 0;
      v14 = (__int64 *)*(v8 - 2);
    }
    else
    {
      if ( v14 == v9 )
      {
        *(v9 - 2) = (__int64)v13;
        *(v9 - 1) = *(v8 - 1);
        *v9 = *v8;
      }
      else
      {
        *(v9 - 2) = (__int64)v13;
        v10 = *v9;
        *(v9 - 1) = *(v8 - 1);
        *v9 = *v8;
        if ( v14 )
        {
          *(v8 - 2) = v14;
          *v8 = v10;
          goto LABEL_8;
        }
      }
      *(v8 - 2) = v8;
      v14 = v8;
    }
LABEL_8:
    *(v8 - 1) = 0;
    *(_BYTE *)v14 = 0;
    v11 = v9[2];
    v12 = v9[4];
    v9[2] = v8[2];
    v9[3] = v8[3];
    v9[4] = v8[4];
    v8[2] = 0;
    v8[3] = 0;
    v8[4] = 0;
    if ( v11 )
      j_j___libc_free_0(v11, v12 - v11);
    v8 += 7;
    v9 += 7;
    --v7;
  }
  while ( v7 );
  v4 = *(char **)(a1 + 8);
  v6 = v4 - a3;
LABEL_18:
  v16 = &a2[v6];
  if ( &a2[v6] != v4 )
  {
    v17 = &a2[v6];
    do
    {
      v18 = *((_QWORD *)v17 + 4);
      if ( v18 )
        j_j___libc_free_0(v18, *((_QWORD *)v17 + 6) - v18);
      if ( *(char **)v17 != v17 + 16 )
        j_j___libc_free_0(*(_QWORD *)v17, *((_QWORD *)v17 + 2) + 1LL);
      v17 += 56;
    }
    while ( v17 != v4 );
    *(_QWORD *)(a1 + 8) = v16;
  }
  return a2;
}
