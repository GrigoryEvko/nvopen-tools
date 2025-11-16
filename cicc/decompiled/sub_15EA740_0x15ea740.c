// Function: sub_15EA740
// Address: 0x15ea740
//
void __fastcall sub_15EA740(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // r15
  __int64 v3; // rcx
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  unsigned __int64 v7; // r13
  _QWORD *v8; // rbx
  unsigned __int64 v9; // r14
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  __int64 v15; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v16; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v17; // [rsp+18h] [rbp-98h]
  __int64 v18; // [rsp+60h] [rbp-50h]
  __int16 v19; // [rsp+70h] [rbp-40h]

  if ( a1 )
  {
    v15 = *(_QWORD *)(a1 + 1552);
    v16 = v15 + 832LL * *(unsigned int *)(a1 + 1560);
    if ( v15 != v16 )
    {
      do
      {
        v16 -= 832LL;
        v17 = *(_QWORD *)(v16 + 40);
        v1 = 192LL * *(unsigned int *)(v16 + 48);
        v2 = v17 + v1;
        if ( v17 != v17 + v1 )
        {
          do
          {
            v3 = *(unsigned int *)(v2 - 120);
            v4 = *(_QWORD *)(v2 - 128);
            v2 -= 192LL;
            v5 = v4 + 56 * v3;
            if ( v4 != v5 )
            {
              do
              {
                v6 = *(unsigned int *)(v5 - 40);
                v7 = *(_QWORD *)(v5 - 48);
                v5 -= 56LL;
                v8 = (_QWORD *)(v7 + 32 * v6);
                if ( (_QWORD *)v7 != v8 )
                {
                  do
                  {
                    v8 -= 4;
                    if ( (_QWORD *)*v8 != v8 + 2 )
                      j_j___libc_free_0(*v8, v8[2] + 1LL);
                  }
                  while ( (_QWORD *)v7 != v8 );
                  v7 = *(_QWORD *)(v5 + 8);
                }
                if ( v7 != v5 + 24 )
                  _libc_free(v7);
              }
              while ( v4 != v5 );
              v4 = *(_QWORD *)(v2 + 64);
            }
            if ( v4 != v2 + 80 )
              _libc_free(v4);
            v9 = *(_QWORD *)(v2 + 16);
            v10 = (_QWORD *)(v9 + 32LL * *(unsigned int *)(v2 + 24));
            if ( (_QWORD *)v9 != v10 )
            {
              do
              {
                v10 -= 4;
                if ( (_QWORD *)*v10 != v10 + 2 )
                  j_j___libc_free_0(*v10, v10[2] + 1LL);
              }
              while ( (_QWORD *)v9 != v10 );
              v9 = *(_QWORD *)(v2 + 16);
            }
            if ( v9 != v2 + 32 )
              _libc_free(v9);
          }
          while ( v17 != v2 );
          v17 = *(_QWORD *)(v16 + 40);
        }
        if ( v17 != v16 + 56 )
          _libc_free(v17);
        if ( *(_QWORD *)v16 != v16 + 16 )
          j_j___libc_free_0(*(_QWORD *)v16, *(_QWORD *)(v16 + 16) + 1LL);
      }
      while ( v15 != v16 );
      v16 = *(_QWORD *)(a1 + 1552);
    }
    if ( v16 != a1 + 1568 )
      _libc_free(v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v13 = (_QWORD *)(a1 + 16);
      v12 = 192;
      LOBYTE(v18) = 0;
      v19 = 257;
    }
    else
    {
      v11 = *(unsigned int *)(a1 + 24);
      if ( !(_DWORD)v11 )
      {
LABEL_44:
        j___libc_free_0(*(_QWORD *)(a1 + 16));
LABEL_42:
        j_j___libc_free_0(a1, 28208);
        return;
      }
      v12 = 6 * v11;
      v13 = *(_QWORD **)(a1 + 16);
      LOBYTE(v18) = 0;
      v19 = 257;
    }
    v14 = &v13[v12];
    do
    {
      if ( (_QWORD *)*v13 != v13 + 2 )
        j_j___libc_free_0(*v13, v13[2] + 1LL);
      v13 += 6;
    }
    while ( v13 != v14 );
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      goto LABEL_42;
    goto LABEL_44;
  }
}
