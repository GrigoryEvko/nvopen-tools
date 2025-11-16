// Function: sub_3382E20
// Address: 0x3382e20
//
__int64 __fastcall sub_3382E20(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx
  _QWORD *v12; // r14
  unsigned __int64 v13; // rdi
  __int64 v14; // rbx
  unsigned __int64 v15; // r15
  __int64 v16; // rsi
  int v17; // ebx
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v20 = a1 + 16;
  v19 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v21, a6);
  v7 = (_QWORD *)v19;
  v8 = *(_QWORD **)a1;
  v9 = 32LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = (_QWORD *)(v19 + v9);
    do
    {
      if ( v7 )
      {
        *v7 = *v8;
        v7[1] = v8[1];
        v7[2] = v8[2];
        v7[3] = v8[3];
        v8[3] = 0;
        v8[2] = 0;
        v8[1] = 0;
      }
      v7 += 4;
      v8 += 4;
    }
    while ( v7 != v11 );
    v12 = *(_QWORD **)a1;
    v10 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v13 = *(_QWORD *)(v10 - 24);
        v14 = *(_QWORD *)(v10 - 16);
        v10 -= 32LL;
        v15 = v13;
        if ( v14 != v13 )
        {
          do
          {
            v16 = *(_QWORD *)(v15 + 24);
            if ( v16 )
              sub_B91220(v15 + 24, v16);
            v15 += 32LL;
          }
          while ( v14 != v15 );
          v13 = *(_QWORD *)(v10 + 8);
        }
        if ( v13 )
          j_j___libc_free_0(v13);
      }
      while ( (_QWORD *)v10 != v12 );
      v10 = *(_QWORD *)a1;
    }
  }
  v17 = v21[0];
  if ( v20 != v10 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v17;
  *(_QWORD *)a1 = v19;
  return v19;
}
