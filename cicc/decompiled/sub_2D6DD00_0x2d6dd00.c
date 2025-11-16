// Function: sub_2D6DD00
// Address: 0x2d6dd00
//
__int64 __fastcall sub_2D6DD00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // r13
  __int64 v10; // r9
  _QWORD *v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // rbx
  unsigned __int64 v16; // r15
  _QWORD *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // ebx
  __int64 v22; // [rsp+0h] [rbp-50h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x428u, v23, a6);
  v9 = *(_QWORD **)a1;
  v22 = v6;
  v10 = 133LL * *(unsigned int *)(a1 + 8);
  v11 = (_QWORD *)(*(_QWORD *)a1 + v10 * 8);
  if ( v9 != &v9[v10] )
  {
    v12 = v6;
    do
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = 0;
        *(_QWORD *)(v12 + 8) = 0;
        v13 = v9[2];
        *(_QWORD *)(v12 + 16) = v13;
        LOBYTE(v8) = v13 != -4096;
        LOBYTE(v7) = v13 != 0;
        if ( ((v13 != 0) & (unsigned __int8)v8) != 0 && v13 != -8192 )
          sub_BD6050((unsigned __int64 *)v12, *v9 & 0xFFFFFFFFFFFFFFF8LL);
        *(_DWORD *)(v12 + 32) = 0;
        *(_QWORD *)(v12 + 24) = v12 + 40;
        *(_DWORD *)(v12 + 36) = 32;
        if ( *((_DWORD *)v9 + 8) )
          sub_2D68580(v12 + 24, (__int64)(v9 + 3), v7, v8);
      }
      v9 += 133;
      v12 += 1064;
    }
    while ( v11 != v9 );
    v14 = *(_QWORD **)a1;
    v11 = (_QWORD *)(*(_QWORD *)a1 + 1064LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v11 )
    {
      do
      {
        v15 = *((unsigned int *)v11 - 258);
        v16 = *(v11 - 130);
        v11 -= 133;
        v17 = (_QWORD *)(v16 + 32 * v15);
        if ( (_QWORD *)v16 != v17 )
        {
          do
          {
            v18 = *(v17 - 2);
            v17 -= 4;
            if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
              sub_BD60C0(v17);
          }
          while ( (_QWORD *)v16 != v17 );
          v16 = v11[3];
        }
        if ( (_QWORD *)v16 != v11 + 5 )
          _libc_free(v16);
        v19 = v11[2];
        if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          sub_BD60C0(v11);
      }
      while ( v11 != v14 );
      v11 = *(_QWORD **)a1;
    }
  }
  v20 = v23[0];
  if ( (_QWORD *)(a1 + 16) != v11 )
    _libc_free((unsigned __int64)v11);
  *(_DWORD *)(a1 + 12) = v20;
  *(_QWORD *)a1 = v22;
  return v22;
}
