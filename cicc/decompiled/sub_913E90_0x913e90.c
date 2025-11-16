// Function: sub_913E90
// Address: 0x913e90
//
__int64 __fastcall sub_913E90(char **a1, char *a2, _QWORD *a3)
{
  char *v4; // r15
  char *v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  char *v8; // r8
  char *v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // r9
  __int64 v13; // rbx
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rbx
  char *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  char *i; // r14
  __int64 v21; // rdx
  __int64 v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  char *v27; // [rsp+18h] [rbp-48h]
  char *v28; // [rsp+20h] [rbp-40h]
  char *v29; // [rsp+20h] [rbp-40h]
  _QWORD *v30; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *a1) >> 3);
  if ( v6 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - v5) >> 3);
  v9 = a2;
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x5555555555555555LL * ((v4 - v5) >> 3);
  v12 = a2 - v5;
  if ( v10 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_34:
    v25 = a3;
    v24 = sub_22077B0(v23);
    v12 = a2 - v5;
    v8 = a2;
    v30 = (_QWORD *)v24;
    a3 = v25;
    v26 = v24 + v23;
    v13 = v24 + 24;
    goto LABEL_7;
  }
  if ( v11 )
  {
    if ( v11 > 0x555555555555555LL )
      v11 = 0x555555555555555LL;
    v23 = 24 * v11;
    goto LABEL_34;
  }
  v26 = 0;
  v13 = 24;
  v30 = 0;
LABEL_7:
  v14 = (_QWORD *)((char *)v30 + v12);
  if ( (_QWORD *)((char *)v30 + v12) )
  {
    v15 = a3[2];
    *v14 = 4;
    v14[1] = 0;
    v14[2] = v15;
    if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
    {
      v28 = v8;
      sub_BD6050(v14, *a3 & 0xFFFFFFFFFFFFFFF8LL);
      v8 = v28;
    }
  }
  if ( v8 != v5 )
  {
    v16 = v30;
    v17 = v5;
    while ( 1 )
    {
      if ( v16 )
      {
        *v16 = 4;
        v16[1] = 0;
        v18 = *((_QWORD *)v17 + 2);
        v16[2] = v18;
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        {
          v27 = v8;
          v29 = v17;
          sub_BD6050(v16, *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL);
          v8 = v27;
          v17 = v29;
        }
      }
      v17 += 24;
      if ( v8 == v17 )
        break;
      v16 += 3;
    }
    v13 = (__int64)(v16 + 6);
  }
  if ( v8 != v4 )
  {
    do
    {
      v19 = *((_QWORD *)v9 + 2);
      *(_QWORD *)v13 = 4;
      *(_QWORD *)(v13 + 8) = 0;
      *(_QWORD *)(v13 + 16) = v19;
      if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
        sub_BD6050(v13, *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL);
      v9 += 24;
      v13 += 24;
    }
    while ( v4 != v9 );
  }
  for ( i = v5; i != v4; i += 24 )
  {
    v21 = *((_QWORD *)i + 2);
    if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
      sub_BD60C0(i);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  a1[1] = (char *)v13;
  *a1 = (char *)v30;
  a1[2] = (char *)v26;
  return v26;
}
