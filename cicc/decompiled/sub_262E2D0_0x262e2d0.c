// Function: sub_262E2D0
// Address: 0x262e2d0
//
void __fastcall sub_262E2D0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  const void *v3; // r15
  __int64 v4; // r12
  unsigned __int64 v5; // r14
  __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // r9
  unsigned __int64 v10; // rcx
  char *v11; // r8
  char *v12; // rax
  _QWORD *v13; // rbx
  char *v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp-48h] [rbp-48h]
  unsigned __int64 v18; // [rsp-40h] [rbp-40h]
  char *v19; // [rsp-40h] [rbp-40h]
  unsigned __int64 v20; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = *(const void **)a1;
    v4 = (__int64)v2 - *(_QWORD *)a1;
    v5 = v4 >> 4;
    if ( a2 > (__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v2) >> 4 )
    {
      if ( 0x7FFFFFFFFFFFFFFLL - v5 < a2 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v6 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4;
      if ( a2 >= v5 )
        v6 = a2;
      v7 = __CFADD__(v5, v6);
      v8 = v5 + v6;
      if ( v7 )
      {
        v15 = 0x7FFFFFFFFFFFFFF0LL;
      }
      else
      {
        if ( !v8 )
        {
          v9 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
          v10 = 0;
          v11 = 0;
LABEL_9:
          v12 = &v11[v4];
          do
          {
            *(_QWORD *)v12 = 0;
            v12 += 16;
            *((_QWORD *)v12 - 1) = 0;
          }
          while ( &v11[16 * a2 + v4] != v12 );
          if ( v9 > 0 )
          {
            v18 = v10;
            v14 = (char *)memmove(v11, v3, v9);
            v10 = v18;
            v11 = v14;
          }
          else if ( !v3 )
          {
LABEL_13:
            *(_QWORD *)a1 = v11;
            *(_QWORD *)(a1 + 16) = v10;
            *(_QWORD *)(a1 + 8) = &v11[16 * v5 + 16 * a2];
            return;
          }
          v17 = v10;
          v19 = v11;
          j_j___libc_free_0((unsigned __int64)v3);
          v10 = v17;
          v11 = v19;
          goto LABEL_13;
        }
        if ( v8 > 0x7FFFFFFFFFFFFFFLL )
          v8 = 0x7FFFFFFFFFFFFFFLL;
        v15 = 16 * v8;
      }
      v20 = v15;
      v16 = sub_22077B0(v15);
      v3 = *(const void **)a1;
      v11 = (char *)v16;
      v10 = v16 + v20;
      v9 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
      goto LABEL_9;
    }
    v13 = &v2[2 * a2];
    do
    {
      *v2 = 0;
      v2 += 2;
      *(v2 - 1) = 0;
    }
    while ( v13 != v2 );
    *(_QWORD *)(a1 + 8) = v13;
  }
}
