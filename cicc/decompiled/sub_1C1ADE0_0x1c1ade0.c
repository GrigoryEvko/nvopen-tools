// Function: sub_1C1ADE0
// Address: 0x1c1ade0
//
__int64 __fastcall sub_1C1ADE0(__int64 a1)
{
  unsigned int v1; // r14d
  __int64 v3; // r12
  unsigned int *v4; // rax
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned int *v12; // rax

  if ( !(unsigned __int8)sub_1C17C50(a1) )
    return 0;
  v3 = sub_22077B0(104);
  if ( !v3 )
  {
    v12 = (unsigned int *)sub_1C17CB0(a1, 0);
    if ( v12 )
      return *v12;
    return 0;
  }
  memset((void *)v3, 0, 0x68u);
  *(_QWORD *)(v3 + 16) = v3 + 32;
  *(_QWORD *)(v3 + 24) = 0x400000000LL;
  *(_QWORD *)(v3 + 64) = v3 + 80;
  *(_QWORD *)(v3 + 88) = 1;
  v4 = (unsigned int *)sub_1C17CB0(a1, (__int64 *)v3);
  if ( v4 )
    v1 = *v4;
  else
    v1 = 0;
  v5 = *(unsigned __int64 **)(v3 + 16);
  v6 = &v5[*(unsigned int *)(v3 + 24)];
  while ( v6 != v5 )
  {
    v7 = *v5++;
    _libc_free(v7);
  }
  v8 = *(unsigned __int64 **)(v3 + 64);
  v9 = (unsigned __int64)&v8[2 * *(unsigned int *)(v3 + 72)];
  if ( v8 != (unsigned __int64 *)v9 )
  {
    do
    {
      v10 = *v8;
      v8 += 2;
      _libc_free(v10);
    }
    while ( v8 != (unsigned __int64 *)v9 );
    v9 = *(_QWORD *)(v3 + 64);
  }
  if ( v3 + 80 != v9 )
    _libc_free(v9);
  v11 = *(_QWORD *)(v3 + 16);
  if ( v3 + 32 != v11 )
    _libc_free(v11);
  j_j___libc_free_0(v3, 104);
  return v1;
}
