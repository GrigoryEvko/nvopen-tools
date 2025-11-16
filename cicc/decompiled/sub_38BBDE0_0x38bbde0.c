// Function: sub_38BBDE0
// Address: 0x38bbde0
//
void __fastcall sub_38BBDE0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 v2; // r13
  __int64 v3; // r14
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r8
  __int64 v9; // r14
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // r14
  unsigned __int64 *v14; // r12
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r12

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_38BBDE0(v1[3]);
      v3 = v1[72];
      v4 = v1[71];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v4 )
      {
        do
        {
          v5 = *(_QWORD *)(v4 + 8);
          if ( v5 )
            j_j___libc_free_0(v5);
          v4 += 32LL;
        }
        while ( v3 != v4 );
        v4 = *(_QWORD *)(v2 + 568);
      }
      if ( v4 )
        j_j___libc_free_0(v4);
      j___libc_free_0(*(_QWORD *)(v2 + 544));
      v6 = *(_QWORD *)(v2 + 456);
      if ( v6 != v2 + 472 )
        j_j___libc_free_0(v6);
      v7 = *(_QWORD *)(v2 + 424);
      if ( v7 != v2 + 440 )
        j_j___libc_free_0(v7);
      v8 = *(_QWORD *)(v2 + 392);
      if ( *(_DWORD *)(v2 + 404) )
      {
        v9 = *(unsigned int *)(v2 + 400);
        if ( (_DWORD)v9 )
        {
          v10 = 8 * v9;
          v11 = 0;
          do
          {
            v12 = *(_QWORD *)(v8 + v11);
            if ( v12 != -8 && v12 )
            {
              _libc_free(v12);
              v8 = *(_QWORD *)(v2 + 392);
            }
            v11 += 8;
          }
          while ( v10 != v11 );
        }
      }
      _libc_free(v8);
      v13 = *(unsigned __int64 **)(v2 + 160);
      v14 = &v13[9 * *(unsigned int *)(v2 + 168)];
      if ( v13 != v14 )
      {
        do
        {
          v14 -= 9;
          if ( (unsigned __int64 *)*v14 != v14 + 2 )
            j_j___libc_free_0(*v14);
        }
        while ( v13 != v14 );
        v14 = *(unsigned __int64 **)(v2 + 160);
      }
      if ( v14 != (unsigned __int64 *)(v2 + 176) )
        _libc_free((unsigned __int64)v14);
      v15 = *(unsigned __int64 **)(v2 + 48);
      v16 = &v15[4 * *(unsigned int *)(v2 + 56)];
      if ( v15 != v16 )
      {
        do
        {
          v16 -= 4;
          if ( (unsigned __int64 *)*v16 != v16 + 2 )
            j_j___libc_free_0(*v16);
        }
        while ( v15 != v16 );
        v16 = *(unsigned __int64 **)(v2 + 48);
      }
      if ( v16 != (unsigned __int64 *)(v2 + 64) )
        _libc_free((unsigned __int64)v16);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
