// Function: sub_2D132C0
// Address: 0x2d132c0
//
void __fastcall sub_2D132C0(char *a1, char *a2, __int64 a3, unsigned __int8 (__fastcall *a4)(__int64, __int64))
{
  __int64 v4; // rbx
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  char *v11; // r14
  char *v12; // rbx
  char *v13; // r15
  __int64 v14; // rsi
  __int64 v15; // rax
  bool v16; // zf
  __int64 v17; // rbx
  __int64 i; // r14
  __int64 *v19; // r15
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rax
  char *v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  char *v25; // [rsp+10h] [rbp-40h]
  char *v26; // [rsp+18h] [rbp-38h]

  v4 = a2 - a1;
  v24 = a3;
  v25 = a2;
  if ( a2 - a1 <= 128 )
    return;
  if ( !a3 )
  {
    v26 = a2;
    goto LABEL_24;
  }
  v23 = a1 + 8;
  while ( 2 )
  {
    --v24;
    v7 = (__int64 *)&a1[8 * ((__int64)(((v25 - a1) >> 3) + ((unsigned __int64)(v25 - a1) >> 63)) >> 1)];
    if ( a4(*((_QWORD *)a1 + 1), *v7) )
    {
      if ( a4(*v7, *((_QWORD *)v25 - 1)) )
      {
        v8 = *(_QWORD *)a1;
        goto LABEL_7;
      }
      if ( a4(*((_QWORD *)a1 + 1), *((_QWORD *)v25 - 1)) )
      {
        v22 = *(_QWORD *)a1;
        *(_QWORD *)a1 = *((_QWORD *)v25 - 1);
        *((_QWORD *)v25 - 1) = v22;
        v9 = *(_QWORD *)a1;
        v10 = *((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_22:
      v10 = *(_QWORD *)a1;
      v9 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
      *(_QWORD *)a1 = v9;
      goto LABEL_8;
    }
    if ( a4(*((_QWORD *)a1 + 1), *((_QWORD *)v25 - 1)) )
      goto LABEL_22;
    v16 = a4(*v7, *((_QWORD *)v25 - 1)) == 0;
    v8 = *(_QWORD *)a1;
    if ( !v16 )
    {
      *(_QWORD *)a1 = *((_QWORD *)v25 - 1);
      *((_QWORD *)v25 - 1) = v8;
      v9 = *(_QWORD *)a1;
      v10 = *((_QWORD *)a1 + 1);
      goto LABEL_8;
    }
LABEL_7:
    *(_QWORD *)a1 = *v7;
    *v7 = v8;
    v9 = *(_QWORD *)a1;
    v10 = *((_QWORD *)a1 + 1);
LABEL_8:
    v11 = v23;
    v12 = v25;
    while ( 1 )
    {
      v26 = v11;
      if ( a4(v10, v9) )
        goto LABEL_14;
      v13 = v12 - 8;
      do
      {
        v14 = *(_QWORD *)v13;
        v12 = v13;
        v13 -= 8;
      }
      while ( a4(*(_QWORD *)a1, v14) );
      if ( v11 >= v12 )
        break;
      v15 = *(_QWORD *)v11;
      *(_QWORD *)v11 = *(_QWORD *)v12;
      *(_QWORD *)v12 = v15;
LABEL_14:
      v10 = *((_QWORD *)v11 + 1);
      v9 = *(_QWORD *)a1;
      v11 += 8;
    }
    v4 = v11 - a1;
    sub_2D132C0(v11, v25, v24, a4);
    if ( v11 - a1 > 128 )
    {
      if ( v24 )
      {
        v25 = v11;
        continue;
      }
LABEL_24:
      v17 = v4 >> 3;
      for ( i = (v17 - 2) >> 1; ; --i )
      {
        sub_2D13140((__int64)a1, i, v17, *(_QWORD *)&a1[8 * i], a4);
        if ( !i )
          break;
      }
      v19 = (__int64 *)(v26 - 8);
      do
      {
        v20 = *v19;
        v21 = (char *)v19-- - a1;
        v19[1] = *(_QWORD *)a1;
        sub_2D13140((__int64)a1, 0, v21 >> 3, v20, a4);
      }
      while ( v21 > 8 );
    }
    break;
  }
}
