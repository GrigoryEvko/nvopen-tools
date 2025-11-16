// Function: sub_1883820
// Address: 0x1883820
//
void __fastcall sub_1883820(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v3; // rax
  const void *v4; // r15
  __int64 v5; // r12
  unsigned __int64 v6; // r14
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // r9
  __int64 v11; // rcx
  char *v12; // r8
  char *v13; // rax
  _QWORD *v14; // rbx
  char *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // [rsp-48h] [rbp-48h]
  __int64 v20; // [rsp-40h] [rbp-40h]
  char *v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v4 = *(const void **)a1;
    v5 = (__int64)v3 - *(_QWORD *)a1;
    v6 = v5 >> 4;
    if ( (__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v3) >> 4 < a2 )
    {
      if ( 0x7FFFFFFFFFFFFFFLL - v6 < a2 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v7 = a2;
      if ( v6 >= a2 )
        v7 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4;
      v8 = __CFADD__(v6, v7);
      v9 = v6 + v7;
      if ( v8 )
      {
        v17 = 0x7FFFFFFFFFFFFFF0LL;
      }
      else
      {
        if ( !v9 )
        {
          v10 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
          v11 = 0;
          v12 = 0;
LABEL_9:
          v13 = &v12[v5];
          do
          {
            *(_QWORD *)v13 = 0;
            v13 += 16;
            *((_QWORD *)v13 - 1) = 0;
          }
          while ( &v12[16 * a2 + v5] != v13 );
          if ( v10 > 0 )
          {
            v20 = v11;
            v15 = (char *)memmove(v12, v4, v10);
            v11 = v20;
            v12 = v15;
            v16 = *(_QWORD *)(a1 + 16) - (_QWORD)v4;
          }
          else
          {
            if ( !v4 )
            {
LABEL_13:
              *(_QWORD *)a1 = v12;
              *(_QWORD *)(a1 + 16) = v11;
              *(_QWORD *)(a1 + 8) = &v12[16 * v6 + 16 * a2];
              return;
            }
            v16 = *(_QWORD *)(a1 + 16) - (_QWORD)v4;
          }
          v19 = v11;
          v21 = v12;
          j_j___libc_free_0(v4, v16);
          v11 = v19;
          v12 = v21;
          goto LABEL_13;
        }
        if ( v9 > 0x7FFFFFFFFFFFFFFLL )
          v9 = 0x7FFFFFFFFFFFFFFLL;
        v17 = 16 * v9;
      }
      v22 = v17;
      v18 = sub_22077B0(v17);
      v4 = *(const void **)a1;
      v12 = (char *)v18;
      v11 = v18 + v22;
      v10 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
      goto LABEL_9;
    }
    v14 = &v3[2 * a2];
    do
    {
      *v3 = 0;
      v3 += 2;
      *(v3 - 1) = 0;
    }
    while ( v14 != v3 );
    *(_QWORD *)(a1 + 8) = v14;
  }
}
