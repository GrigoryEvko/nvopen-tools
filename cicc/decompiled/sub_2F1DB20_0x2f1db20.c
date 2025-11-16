// Function: sub_2F1DB20
// Address: 0x2f1db20
//
void __fastcall sub_2F1DB20(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  _QWORD *v19; // [rsp+28h] [rbp-38h]

  if ( !a2 )
    return;
  v2 = *(_QWORD **)a1;
  v19 = *(_QWORD **)(a1 + 8);
  v3 = (__int64)v19 - *(_QWORD *)a1;
  v17 = v3 >> 5;
  if ( a2 <= (__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v19) >> 5 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = a2;
    do
    {
      if ( v4 )
      {
        *(_QWORD *)(v4 + 16) = 0;
        *(_OWORD *)v4 = 0;
        *(_QWORD *)(v4 + 24) = 0;
        *(_QWORD *)(v4 + 8) = 0;
      }
      v4 += 32;
      --v5;
    }
    while ( v5 );
    *(_QWORD *)(a1 + 8) = &v19[4 * a2];
    return;
  }
  if ( 0x3FFFFFFFFFFFFFFLL - v17 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v6 = ((__int64)v19 - *(_QWORD *)a1) >> 5;
  if ( a2 >= v17 )
    v6 = a2;
  v7 = __CFADD__(v17, v6);
  v8 = v17 + v6;
  if ( v7 )
  {
    v14 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_38:
    v15 = sub_22077B0(v14);
    v2 = *(_QWORD **)a1;
    v18 = v15;
    v19 = *(_QWORD **)(a1 + 8);
    v16 = v15 + v14;
    goto LABEL_15;
  }
  if ( v8 )
  {
    if ( v8 > 0x3FFFFFFFFFFFFFFLL )
      v8 = 0x3FFFFFFFFFFFFFFLL;
    v14 = 32 * v8;
    goto LABEL_38;
  }
  v16 = 0;
  v18 = 0;
LABEL_15:
  v9 = a2;
  v10 = v3 + v18;
  do
  {
    if ( v10 )
    {
      *(_QWORD *)(v10 + 16) = 0;
      *(_OWORD *)v10 = 0;
      *(_QWORD *)(v10 + 24) = 0;
      *(_QWORD *)(v10 + 8) = 0;
    }
    v10 += 32;
    --v9;
  }
  while ( v9 );
  if ( v19 != v2 )
  {
    v11 = v18;
    while ( 1 )
    {
      while ( v11 )
      {
        *(_QWORD *)v11 = *v2;
        *(_QWORD *)(v11 + 8) = v2[1];
        *(_QWORD *)(v11 + 16) = v2[2];
        *(_QWORD *)(v11 + 24) = v2[3];
        v2[3] = 0;
        v2[2] = 0;
        v2[1] = 0;
LABEL_22:
        v2 += 4;
        v11 += 32;
        if ( v2 == v19 )
          goto LABEL_31;
      }
      v12 = (unsigned __int64 *)v2[2];
      v13 = (unsigned __int64 *)v2[1];
      if ( v12 != v13 )
      {
        do
        {
          if ( (unsigned __int64 *)*v13 != v13 + 2 )
            j_j___libc_free_0(*v13);
          v13 += 7;
        }
        while ( v12 != v13 );
        v13 = (unsigned __int64 *)v2[1];
      }
      if ( !v13 )
        goto LABEL_22;
      v2 += 4;
      v11 = 32;
      j_j___libc_free_0((unsigned __int64)v13);
      if ( v2 == v19 )
      {
LABEL_31:
        v2 = *(_QWORD **)a1;
        break;
      }
    }
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  *(_QWORD *)a1 = v18;
  *(_QWORD *)(a1 + 8) = 32 * (v17 + a2) + v18;
  *(_QWORD *)(a1 + 16) = v16;
}
