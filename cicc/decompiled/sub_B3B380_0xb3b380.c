// Function: sub_B3B380
// Address: 0xb3b380
//
void __fastcall sub_B3B380(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // rcx
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rbx
  _QWORD *v8; // r14
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  _QWORD *v11; // rbx
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // r12
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // [rsp+0h] [rbp-B0h]
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+60h] [rbp-50h]
  __int16 v21; // [rsp+70h] [rbp-40h]

  if ( a1 )
  {
    v1 = *(_QWORD *)(a1 + 1552);
    v17 = v1;
    v18 = v1 + 832LL * *(unsigned int *)(a1 + 1560);
    if ( v1 != v18 )
    {
      do
      {
        v18 -= 832;
        v19 = *(_QWORD *)(v18 + 40);
        v2 = 192LL * *(unsigned int *)(v18 + 48);
        v3 = v19 + v2;
        if ( v19 != v19 + v2 )
        {
          do
          {
            v4 = *(unsigned int *)(v3 - 120);
            v5 = *(_QWORD *)(v3 - 128);
            v3 -= 192;
            v6 = v5 + 56 * v4;
            if ( v5 != v6 )
            {
              do
              {
                v7 = *(unsigned int *)(v6 - 40);
                v8 = *(_QWORD **)(v6 - 48);
                v6 -= 56;
                v9 = &v8[4 * v7];
                if ( v8 != v9 )
                {
                  do
                  {
                    v9 -= 4;
                    if ( (_QWORD *)*v9 != v9 + 2 )
                    {
                      v1 = v9[2] + 1LL;
                      j_j___libc_free_0(*v9, v1);
                    }
                  }
                  while ( v8 != v9 );
                  v8 = *(_QWORD **)(v6 + 8);
                }
                if ( v8 != (_QWORD *)(v6 + 24) )
                  _libc_free(v8, v1);
              }
              while ( v5 != v6 );
              v5 = *(_QWORD *)(v3 + 64);
            }
            if ( v5 != v3 + 80 )
              _libc_free(v5, v1);
            v10 = *(_QWORD **)(v3 + 16);
            v11 = &v10[4 * *(unsigned int *)(v3 + 24)];
            if ( v10 != v11 )
            {
              do
              {
                v11 -= 4;
                if ( (_QWORD *)*v11 != v11 + 2 )
                {
                  v1 = v11[2] + 1LL;
                  j_j___libc_free_0(*v11, v1);
                }
              }
              while ( v10 != v11 );
              v10 = *(_QWORD **)(v3 + 16);
            }
            if ( v10 != (_QWORD *)(v3 + 32) )
              _libc_free(v10, v1);
          }
          while ( v19 != v3 );
          v19 = *(_QWORD *)(v18 + 40);
        }
        if ( v19 != v18 + 56 )
          _libc_free(v19, v1);
        v1 = v18;
        if ( *(_QWORD *)v18 != v18 + 16 )
        {
          v1 = *(_QWORD *)(v18 + 16) + 1LL;
          j_j___libc_free_0(*(_QWORD *)v18, v1);
        }
      }
      while ( v17 != v18 );
      v18 = *(_QWORD *)(a1 + 1552);
    }
    if ( v18 != a1 + 1568 )
      _libc_free(v18, v1);
    v12 = *(_BYTE *)(a1 + 8);
    if ( (v12 & 1) != 0 )
    {
      v14 = 192;
      v15 = (_QWORD *)(a1 + 16);
      LOBYTE(v20) = 0;
      v21 = 257;
    }
    else
    {
      v13 = *(unsigned int *)(a1 + 24);
      v14 = 6 * v13;
      if ( !(_DWORD)v13 )
        goto LABEL_46;
      v15 = *(_QWORD **)(a1 + 16);
      LOBYTE(v20) = 0;
      v21 = 257;
    }
    v16 = &v15[v14];
    if ( v15 != v16 )
    {
      do
      {
        if ( (_QWORD *)*v15 != v15 + 2 )
          j_j___libc_free_0(*v15, v15[2] + 1LL);
        v15 += 6;
      }
      while ( v15 != v16 );
      v12 = *(_BYTE *)(a1 + 8);
    }
    if ( (v12 & 1) != 0 )
      goto LABEL_43;
    v13 = *(unsigned int *)(a1 + 24);
LABEL_46:
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 48 * v13, 8);
LABEL_43:
    j_j___libc_free_0(a1, 28208);
  }
}
