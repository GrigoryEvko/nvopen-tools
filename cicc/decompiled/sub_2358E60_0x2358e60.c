// Function: sub_2358E60
// Address: 0x2358e60
//
__int64 __fastcall sub_2358E60(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  __int64 v8; // rcx
  unsigned __int64 v9; // r12
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx
  _QWORD *v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r15
  int v16; // ebx
  __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v19 = a1 + 16;
  v18 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v20, a6);
  v7 = *(_QWORD **)a1;
  v8 = 32LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)(v18 + v8);
    v11 = (_QWORD *)v18;
    do
    {
      if ( v11 )
      {
        *v11 = *v7;
        v11[1] = v7[1];
        v11[2] = v7[2];
        v11[3] = v7[3];
        v7[3] = 0;
        v7[2] = 0;
        v7[1] = 0;
      }
      v11 += 4;
      v7 += 4;
    }
    while ( v11 != v10 );
    v12 = *(_QWORD **)a1;
    v9 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v13 = *(_QWORD *)(v9 - 24);
        v14 = *(unsigned __int64 **)(v9 - 16);
        v9 -= 32LL;
        v15 = (unsigned __int64 *)v13;
        if ( v14 != (unsigned __int64 *)v13 )
        {
          do
          {
            if ( (unsigned __int64 *)*v15 != v15 + 2 )
              _libc_free(*v15);
            v15 += 21;
          }
          while ( v14 != v15 );
          v13 = *(_QWORD *)(v9 + 8);
        }
        if ( v13 )
          j_j___libc_free_0(v13);
      }
      while ( (_QWORD *)v9 != v12 );
      v9 = *(_QWORD *)a1;
    }
  }
  v16 = v20[0];
  if ( v19 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v16;
  *(_QWORD *)a1 = v18;
  return v18;
}
