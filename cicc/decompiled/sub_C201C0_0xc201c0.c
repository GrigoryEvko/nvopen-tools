// Function: sub_C201C0
// Address: 0xc201c0
//
__int64 __fastcall sub_C201C0(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // rdi
  __int64 v3; // rdi
  _QWORD *v4; // r12
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // r15
  __int64 v13; // r13
  _QWORD *v14; // rdi
  __int64 v15; // rdi
  __int64 result; // rax
  _QWORD *v18; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD **)(a1 + 120);
  *(_QWORD *)a1 = &unk_49DBA88;
  while ( v1 )
  {
    v2 = v1;
    v1 = (_QWORD *)*v1;
    j_j___libc_free_0(v2, 32);
  }
  memset(*(void **)(a1 + 104), 0, 8LL * *(_QWORD *)(a1 + 112));
  v3 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  if ( v3 != a1 + 152 )
    j_j___libc_free_0(v3, 8LL * *(_QWORD *)(a1 + 112));
  v4 = *(_QWORD **)(a1 + 88);
  if ( v4 )
  {
    sub_C7D6A0(v4[3], 24LL * *((unsigned int *)v4 + 10), 8);
    v5 = v4[1];
    if ( v5 )
    {
      sub_EE5E50(v4[1]);
      j_j___libc_free_0(v5, 8);
    }
    if ( *v4 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
    j_j___libc_free_0(v4, 64);
  }
  v6 = *(_QWORD *)(a1 + 80);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 8);
    if ( v7 )
      j_j___libc_free_0(v7, *(_QWORD *)(v6 + 24) - v7);
    j_j___libc_free_0(v6, 88);
  }
  v8 = *(_QWORD *)(a1 + 72);
  if ( v8 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  v18 = *(_QWORD **)(a1 + 24);
  while ( v18 )
  {
    v9 = v18;
    v10 = v18[19];
    v18 = (_QWORD *)*v18;
    while ( v10 )
    {
      v11 = v10;
      sub_C1F230(*(_QWORD **)(v10 + 24));
      v12 = *(_QWORD *)(v10 + 56);
      v10 = *(_QWORD *)(v10 + 16);
      while ( v12 )
      {
        v13 = v12;
        sub_C1F480(*(_QWORD **)(v12 + 24));
        v14 = *(_QWORD **)(v12 + 184);
        v12 = *(_QWORD *)(v12 + 16);
        sub_C1F230(v14);
        sub_C1EF60(*(_QWORD **)(v13 + 136));
        j_j___libc_free_0(v13, 224);
      }
      j_j___libc_free_0(v11, 88);
    }
    sub_C1EF60((_QWORD *)v9[13]);
    j_j___libc_free_0(v9, 200);
  }
  memset(*(void **)(a1 + 8), 0, 8LL * *(_QWORD *)(a1 + 16));
  v15 = *(_QWORD *)(a1 + 8);
  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v15 != a1 + 56 )
    return j_j___libc_free_0(v15, 8 * result);
  return result;
}
