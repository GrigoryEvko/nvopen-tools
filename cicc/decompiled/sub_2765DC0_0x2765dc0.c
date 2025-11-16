// Function: sub_2765DC0
// Address: 0x2765dc0
//
void __fastcall sub_2765DC0(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  int v4; // eax
  unsigned __int64 v5; // r14
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r12
  unsigned __int64 v12; // rcx
  int v13; // eax
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rcx
  unsigned int v17; // ecx
  const void **v18; // rsi
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // [rsp+10h] [rbp-40h]
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  if ( v3 != a1[2] )
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *(_QWORD *)a2;
      v4 = *(_DWORD *)(a2 + 16);
      *(_DWORD *)(a2 + 16) = 0;
      *(_DWORD *)(v3 + 16) = v4;
      *(_QWORD *)(v3 + 8) = *(_QWORD *)(a2 + 8);
      v3 = a1[1];
    }
    a1[1] = v3 + 24;
    return;
  }
  v5 = *a1;
  v6 = v3 - *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 - v5) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((__int64)(v3 - v5) >> 3);
  if ( v9 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v23 = 0;
      v11 = 24;
      v24 = 0;
      goto LABEL_11;
    }
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v21 = 24 * v10;
  }
  v22 = sub_22077B0(v21);
  v6 = v3 - v5;
  v24 = v22;
  v11 = v22 + 24;
  v23 = v22 + v21;
LABEL_11:
  v12 = v24 + v6;
  if ( v12 )
  {
    *(_QWORD *)v12 = *(_QWORD *)a2;
    v13 = *(_DWORD *)(a2 + 16);
    *(_DWORD *)(a2 + 16) = 0;
    *(_DWORD *)(v12 + 16) = v13;
    *(_QWORD *)(v12 + 8) = *(_QWORD *)(a2 + 8);
  }
  if ( v3 != v5 )
  {
    v14 = v24;
    v15 = v5;
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_16;
      *(_QWORD *)v14 = *(_QWORD *)v15;
      v17 = *(_DWORD *)(v15 + 16);
      *(_DWORD *)(v14 + 16) = v17;
      if ( v17 <= 0x40 )
        break;
      v18 = (const void **)(v15 + 8);
      v15 += 24LL;
      sub_C43780(v14 + 8, v18);
      v16 = v14 + 24;
      if ( v3 == v15 )
      {
LABEL_21:
        v11 = v14 + 48;
        v19 = v5;
        do
        {
          if ( *(_DWORD *)(v19 + 16) > 0x40u )
          {
            v20 = *(_QWORD *)(v19 + 8);
            if ( v20 )
              j_j___libc_free_0_0(v20);
          }
          v19 += 24LL;
        }
        while ( v3 != v19 );
        goto LABEL_26;
      }
LABEL_17:
      v14 = v16;
    }
    *(_QWORD *)(v14 + 8) = *(_QWORD *)(v15 + 8);
LABEL_16:
    v15 += 24LL;
    v16 = v14 + 24;
    if ( v3 == v15 )
      goto LABEL_21;
    goto LABEL_17;
  }
LABEL_26:
  if ( v5 )
    j_j___libc_free_0(v5);
  a1[1] = v11;
  *a1 = v24;
  a1[2] = v23;
}
