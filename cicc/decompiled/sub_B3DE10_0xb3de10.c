// Function: sub_B3DE10
// Address: 0xb3de10
//
__int64 __fastcall sub_B3DE10(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rbx
  _QWORD *v12; // r14
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  __int64 v16; // [rsp+8h] [rbp-38h]

  result = *((unsigned int *)a1 + 2);
  v3 = *a1;
  v4 = *a1 + 192 * result;
  if ( *a1 != v4 )
  {
    v5 = a2;
    do
    {
      if ( v5 )
      {
        *(_DWORD *)v5 = *(_DWORD *)v3;
        *(_DWORD *)(v5 + 4) = *(_DWORD *)(v3 + 4);
        *(_BYTE *)(v5 + 8) = *(_BYTE *)(v3 + 8);
        *(_BYTE *)(v5 + 9) = *(_BYTE *)(v3 + 9);
        *(_BYTE *)(v5 + 10) = *(_BYTE *)(v3 + 10);
        *(_BYTE *)(v5 + 11) = *(_BYTE *)(v3 + 11);
        v6 = *(_DWORD *)(v3 + 12);
        *(_DWORD *)(v5 + 24) = 0;
        *(_DWORD *)(v5 + 12) = v6;
        *(_QWORD *)(v5 + 16) = v5 + 32;
        *(_DWORD *)(v5 + 28) = 1;
        if ( *(_DWORD *)(v3 + 24) )
        {
          a2 = v3 + 16;
          sub_B3BE00(v5 + 16, v3 + 16);
        }
        *(_DWORD *)(v5 + 72) = 0;
        *(_QWORD *)(v5 + 64) = v5 + 80;
        *(_DWORD *)(v5 + 76) = 2;
        if ( *(_DWORD *)(v3 + 72) )
        {
          a2 = v3 + 64;
          sub_B3D620(v5 + 64, v3 + 64);
        }
      }
      v3 += 192;
      v5 += 192;
    }
    while ( v4 != v3 );
    result = *((unsigned int *)a1 + 2);
    v16 = *a1;
    v7 = *a1 + 192 * result;
    if ( *a1 != v7 )
    {
      do
      {
        v8 = *(unsigned int *)(v7 - 120);
        v9 = *(_QWORD *)(v7 - 128);
        v7 -= 192;
        v10 = v9 + 56 * v8;
        if ( v9 != v10 )
        {
          do
          {
            v11 = *(unsigned int *)(v10 - 40);
            v12 = *(_QWORD **)(v10 - 48);
            v10 -= 56;
            v13 = &v12[4 * v11];
            if ( v12 != v13 )
            {
              do
              {
                v13 -= 4;
                if ( (_QWORD *)*v13 != v13 + 2 )
                {
                  a2 = v13[2] + 1LL;
                  j_j___libc_free_0(*v13, a2);
                }
              }
              while ( v12 != v13 );
              v12 = *(_QWORD **)(v10 + 8);
            }
            if ( v12 != (_QWORD *)(v10 + 24) )
              _libc_free(v12, a2);
          }
          while ( v9 != v10 );
          v9 = *(_QWORD *)(v7 + 64);
        }
        if ( v9 != v7 + 80 )
          _libc_free(v9, a2);
        v14 = *(_QWORD **)(v7 + 16);
        v15 = &v14[4 * *(unsigned int *)(v7 + 24)];
        if ( v14 != v15 )
        {
          do
          {
            v15 -= 4;
            if ( (_QWORD *)*v15 != v15 + 2 )
            {
              a2 = v15[2] + 1LL;
              j_j___libc_free_0(*v15, a2);
            }
          }
          while ( v14 != v15 );
          v14 = *(_QWORD **)(v7 + 16);
        }
        result = v7 + 32;
        if ( v14 != (_QWORD *)(v7 + 32) )
          result = _libc_free(v14, a2);
      }
      while ( v7 != v16 );
    }
  }
  return result;
}
