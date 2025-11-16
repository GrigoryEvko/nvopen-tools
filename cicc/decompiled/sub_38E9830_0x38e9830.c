// Function: sub_38E9830
// Address: 0x38e9830
//
void __fastcall sub_38E9830(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r8
  unsigned __int64 *v3; // r15
  unsigned __int64 v4; // rdx
  __int64 v6; // rbx
  _QWORD *v7; // rax
  unsigned __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+8h] [rbp-58h]
  unsigned __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  unsigned __int64 v22; // [rsp+18h] [rbp-48h]
  unsigned __int64 v23; // [rsp+28h] [rbp-38h]

  if ( !a2 )
    return;
  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(unsigned __int64 **)a1;
  v4 = a2;
  v23 = v2;
  v6 = v2 - *(_QWORD *)a1;
  v22 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - v2) >> 3) >= a2 )
  {
    v7 = *(_QWORD **)(a1 + 8);
    do
    {
      if ( v7 )
      {
        *v7 = 0;
        v7[1] = 0;
        v7[2] = 0;
      }
      v7 += 3;
      --v4;
    }
    while ( v4 );
    *(_QWORD *)(a1 + 8) = v2 + 24 * a2;
    return;
  }
  if ( 0x555555555555555LL - v22 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = a2;
  if ( v22 >= a2 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v2 - *(_QWORD *)a1) >> 3);
  v9 = __CFADD__(v22, v8);
  v10 = v22 + v8;
  if ( v9 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_39:
    v20 = v17;
    v18 = sub_22077B0(v17);
    v3 = *(unsigned __int64 **)a1;
    v21 = v18;
    v23 = *(_QWORD *)(a1 + 8);
    v19 = v18 + v20;
    goto LABEL_15;
  }
  if ( v10 )
  {
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v17 = 24 * v10;
    goto LABEL_39;
  }
  v19 = 0;
  v21 = 0;
LABEL_15:
  v11 = a2;
  v12 = (_QWORD *)(v6 + v21);
  do
  {
    if ( v12 )
    {
      *v12 = 0;
      v12[1] = 0;
      v12[2] = 0;
    }
    v12 += 3;
    --v11;
  }
  while ( v11 );
  if ( v3 != (unsigned __int64 *)v23 )
  {
    v13 = v21;
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = v3[1];
        v15 = *v3;
        if ( !v13 )
          break;
        *(_QWORD *)v13 = v15;
        *(_QWORD *)(v13 + 8) = v14;
        *(_QWORD *)(v13 + 16) = v3[2];
        v3[2] = 0;
        v3[1] = 0;
        *v3 = 0;
LABEL_22:
        v3 += 3;
        v13 += 24;
        if ( v3 == (unsigned __int64 *)v23 )
          goto LABEL_32;
      }
      if ( v14 != v15 )
      {
        do
        {
          if ( *(_DWORD *)(v15 + 32) > 0x40u )
          {
            v16 = *(_QWORD *)(v15 + 24);
            if ( v16 )
              j_j___libc_free_0_0(v16);
          }
          v15 += 40LL;
        }
        while ( v15 != v14 );
        v15 = *v3;
      }
      if ( !v15 )
        goto LABEL_22;
      v3 += 3;
      v13 = 24;
      j_j___libc_free_0(v15);
      if ( v3 == (unsigned __int64 *)v23 )
      {
LABEL_32:
        v23 = *(_QWORD *)a1;
        break;
      }
    }
  }
  if ( v23 )
    j_j___libc_free_0(v23);
  *(_QWORD *)a1 = v21;
  *(_QWORD *)(a1 + 8) = v21 + 24 * (a2 + v22);
  *(_QWORD *)(a1 + 16) = v19;
}
