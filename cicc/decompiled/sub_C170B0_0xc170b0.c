// Function: sub_C170B0
// Address: 0xc170b0
//
__int64 __fastcall sub_C170B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 *v5; // rax
  __int64 *v6; // r12
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r15
  _QWORD *v11; // r14
  int v12; // ebx
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 *v15; // [rsp+10h] [rbp-50h]
  __int64 *v16; // [rsp+18h] [rbp-48h]
  _QWORD v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v15 = (__int64 *)(a1 + 16);
  v14 = sub_C8D7D0(a1, a1 + 16, a2, 24, v17);
  v5 = *(__int64 **)a1;
  v6 = (__int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
  if ( *(__int64 **)a1 != v6 )
  {
    v7 = (_QWORD *)v14;
    do
    {
      if ( v7 )
      {
        *v7 = *v5;
        v7[1] = v5[1];
        v7[2] = v5[2];
        v5[2] = 0;
        v5[1] = 0;
        *v5 = 0;
      }
      v5 += 3;
      v7 += 3;
    }
    while ( v6 != v5 );
    v16 = *(__int64 **)a1;
    v6 = (__int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(__int64 **)a1 != v6 )
    {
      do
      {
        v8 = *(v6 - 3);
        v9 = *(v6 - 2);
        v6 -= 3;
        v10 = v8;
        if ( v9 != v8 )
        {
          do
          {
            v11 = *(_QWORD **)(v10 + 8);
            if ( v11 )
            {
              if ( (_QWORD *)*v11 != v11 + 2 )
                j_j___libc_free_0(*v11, v11[2] + 1LL);
              v3 = 32;
              j_j___libc_free_0(v11, 32);
            }
            v10 += 32;
          }
          while ( v9 != v10 );
          v8 = *v6;
        }
        if ( v8 )
        {
          v3 = v6[2] - v8;
          j_j___libc_free_0(v8, v3);
        }
      }
      while ( v6 != v16 );
      v6 = *(__int64 **)a1;
    }
  }
  v12 = v17[0];
  if ( v15 != v6 )
    _libc_free(v6, v3);
  *(_DWORD *)(a1 + 12) = v12;
  *(_QWORD *)a1 = v14;
  return v14;
}
