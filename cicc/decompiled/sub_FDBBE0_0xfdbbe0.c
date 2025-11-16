// Function: sub_FDBBE0
// Address: 0xfdbbe0
//
void *__fastcall sub_FDBBE0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdi
  _QWORD *v5; // rbx
  void *result; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  _QWORD *v10; // rdi
  __int64 v11; // rdi
  _QWORD *i; // r12
  _QWORD *v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD v20[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v21; // [rsp+10h] [rbp-50h]
  _QWORD v22[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+30h] [rbp-30h]

  *(_QWORD *)a1 = &unk_49E5470;
  v2 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v2 )
  {
    v15 = *(_QWORD **)(a1 + 168);
    v20[0] = 0;
    v20[1] = 0;
    v16 = &v15[9 * v2];
    v21 = -4096;
    v17 = -4096;
    v22[0] = 0;
    v22[1] = 0;
    v23 = -8192;
    while ( 1 )
    {
      v18 = v15[2];
      if ( v18 != v17 )
      {
        v17 = v23;
        if ( v18 != v23 )
        {
          v15[4] = &unk_49DB368;
          v19 = v15[7];
          if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
          {
            sub_BD60C0(v15 + 5);
            v18 = v15[2];
          }
          v17 = v18;
        }
      }
      if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        sub_BD60C0(v15);
      v15 += 9;
      if ( v16 == v15 )
        break;
      v17 = v21;
    }
    if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
      sub_BD60C0(v22);
    if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
      sub_BD60C0(v20);
    v2 = *(unsigned int *)(a1 + 184);
  }
  v3 = 72 * v2;
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 72 * v2, 8);
  v4 = *(_QWORD *)(a1 + 136);
  if ( v4 )
  {
    v3 = *(_QWORD *)(a1 + 152) - v4;
    j_j___libc_free_0(v4, v3);
  }
  v5 = *(_QWORD **)(a1 + 88);
  result = &unk_49E5580;
  for ( *(_QWORD *)a1 = &unk_49E5580; (_QWORD *)(a1 + 88) != v5; result = (void *)j_j___libc_free_0(v7, 192) )
  {
    v7 = v5;
    v5 = (_QWORD *)*v5;
    v8 = (_QWORD *)v7[18];
    if ( v8 != v7 + 20 )
      _libc_free(v8, v3);
    v9 = (_QWORD *)v7[14];
    if ( v9 != v7 + 16 )
      _libc_free(v9, v3);
    v10 = (_QWORD *)v7[4];
    if ( v10 != v7 + 6 )
      _libc_free(v10, v3);
    v3 = 192;
  }
  v11 = *(_QWORD *)(a1 + 64);
  if ( v11 )
    result = (void *)j_j___libc_free_0(v11, *(_QWORD *)(a1 + 80) - v11);
  for ( i = *(_QWORD **)(a1 + 32); (_QWORD *)(a1 + 32) != i; result = (void *)j_j___libc_free_0(v13, 40) )
  {
    v13 = i;
    i = (_QWORD *)*i;
  }
  v14 = *(_QWORD *)(a1 + 8);
  if ( v14 )
    return (void *)j_j___libc_free_0(v14, *(_QWORD *)(a1 + 24) - v14);
  return result;
}
