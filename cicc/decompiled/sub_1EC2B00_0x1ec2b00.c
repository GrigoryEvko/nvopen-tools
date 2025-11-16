// Function: sub_1EC2B00
// Address: 0x1ec2b00
//
__int64 __fastcall sub_1EC2B00(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r15
  __int64 v9; // rdi
  __int64 v10; // rcx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r9
  unsigned __int64 *v13; // rbx
  __int64 v14; // rax
  unsigned __int64 *v15; // r13
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi

  v1 = a1 - 232;
  v2 = a1 + 1096;
  do
  {
    v4 = *(_QWORD *)(v2 + 136);
    if ( v4 != v2 + 152 )
      _libc_free(v4);
    v5 = *(_QWORD *)(v2 + 96);
    if ( v5 != v2 + 112 )
      _libc_free(v5);
    v6 = *(unsigned int *)(v2 + 88);
    if ( (_DWORD)v6 )
    {
      v7 = *(_QWORD **)(v2 + 72);
      v8 = &v7[7 * v6];
      do
      {
        if ( *v7 != -16 && *v7 != -8 )
        {
          _libc_free(v7[4]);
          _libc_free(v7[1]);
        }
        v7 += 7;
      }
      while ( v8 != v7 );
    }
    v9 = *(_QWORD *)(v2 + 72);
    v2 -= 664;
    j___libc_free_0(v9);
    _libc_free(*(_QWORD *)(v2 + 704));
  }
  while ( v1 != v2 );
  j___libc_free_0(*(_QWORD *)(a1 + 408));
  if ( *(_DWORD *)(a1 + 384) )
    sub_1EC25D0(a1 + 200, (char *)sub_1EBAFD0, 0, v10, v11, v12);
  v13 = *(unsigned __int64 **)(a1 + 112);
  v14 = *(unsigned int *)(a1 + 120);
  *(_QWORD *)(a1 + 88) = 0;
  v15 = &v13[v14];
  while ( v15 != v13 )
  {
    v16 = *v13++;
    _libc_free(v16);
  }
  v17 = *(unsigned __int64 **)(a1 + 160);
  v18 = (unsigned __int64)&v17[2 * *(unsigned int *)(a1 + 168)];
  if ( v17 != (unsigned __int64 *)v18 )
  {
    do
    {
      v19 = *v17;
      v17 += 2;
      _libc_free(v19);
    }
    while ( (unsigned __int64 *)v18 != v17 );
    v18 = *(_QWORD *)(a1 + 160);
  }
  if ( v18 != a1 + 176 )
    _libc_free(v18);
  v20 = *(_QWORD *)(a1 + 112);
  if ( v20 != a1 + 128 )
    _libc_free(v20);
  return j_j___libc_free_0(a1, 1760);
}
