// Function: sub_38E1100
// Address: 0x38e1100
//
__int64 __fastcall sub_38E1100(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // r15
  unsigned int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rcx
  char v10; // di
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r9
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r14
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r13
  unsigned int v23; // [rsp+14h] [rbp-3Ch]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v5 = sub_22077B0(0x250u);
  v6 = **a3;
  *(_DWORD *)(v5 + 32) = v6;
  memset((void *)(v5 + 40), 0, 0x228u);
  v23 = v6;
  *(_QWORD *)(v5 + 56) = 0x300000000LL;
  *(_QWORD *)(v5 + 168) = 0x300000000LL;
  *(_QWORD *)(v5 + 408) = 0x1000000000LL;
  *(_QWORD *)(v5 + 48) = v5 + 64;
  *(_QWORD *)(v5 + 160) = v5 + 176;
  *(_QWORD *)(v5 + 424) = v5 + 440;
  *(_QWORD *)(v5 + 456) = v5 + 472;
  *(_BYTE *)(v5 + 529) = 1;
  v7 = sub_38C3D00(a1, a2, (unsigned int *)(v5 + 32));
  v24 = v7;
  if ( v8 )
  {
    v9 = a1 + 1;
    v10 = 1;
    if ( !v7 && (_QWORD *)v8 != v9 )
      v10 = v23 < *(_DWORD *)(v8 + 32);
    sub_220F040(v10, v5, (_QWORD *)v8, v9);
    ++a1[5];
    return v5;
  }
  else
  {
    j___libc_free_0(0);
    v12 = *(_QWORD *)(v5 + 456);
    if ( v5 + 472 != v12 )
      j_j___libc_free_0(v12);
    v13 = *(_QWORD *)(v5 + 424);
    if ( v5 + 440 != v13 )
      j_j___libc_free_0(v13);
    v14 = *(_QWORD *)(v5 + 392);
    if ( *(_DWORD *)(v5 + 404) )
    {
      v15 = *(unsigned int *)(v5 + 400);
      if ( (_DWORD)v15 )
      {
        v16 = 8 * v15;
        v17 = 0;
        do
        {
          v18 = *(_QWORD *)(v14 + v17);
          if ( v18 != -8 && v18 )
          {
            _libc_free(v18);
            v14 = *(_QWORD *)(v5 + 392);
          }
          v17 += 8;
        }
        while ( v16 != v17 );
      }
    }
    _libc_free(v14);
    v19 = *(unsigned __int64 **)(v5 + 160);
    v20 = &v19[9 * *(unsigned int *)(v5 + 168)];
    if ( v19 != v20 )
    {
      do
      {
        v20 -= 9;
        if ( (unsigned __int64 *)*v20 != v20 + 2 )
          j_j___libc_free_0(*v20);
      }
      while ( v19 != v20 );
      v20 = *(unsigned __int64 **)(v5 + 160);
    }
    if ( (unsigned __int64 *)(v5 + 176) != v20 )
      _libc_free((unsigned __int64)v20);
    v21 = *(unsigned __int64 **)(v5 + 48);
    v22 = &v21[4 * *(unsigned int *)(v5 + 56)];
    if ( v21 != v22 )
    {
      do
      {
        v22 -= 4;
        if ( (unsigned __int64 *)*v22 != v22 + 2 )
          j_j___libc_free_0(*v22);
      }
      while ( v21 != v22 );
      v22 = *(unsigned __int64 **)(v5 + 48);
    }
    if ( (unsigned __int64 *)(v5 + 64) != v22 )
      _libc_free((unsigned __int64)v22);
    j_j___libc_free_0(v5);
    return v24;
  }
}
