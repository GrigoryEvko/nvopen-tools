// Function: sub_239A190
// Address: 0x239a190
//
__int64 __fastcall sub_239A190(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // rdx
  _QWORD *v11; // rcx
  _QWORD *v12; // rcx
  _QWORD *v13; // rsi
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdi
  _QWORD *v16; // r14
  unsigned __int64 v17; // rdi
  int v18; // r13d
  __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (_QWORD *)(a1 + 16);
  v20 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v21, a6);
  v8 = *(_QWORD **)a1;
  v9 = (_QWORD *)(*(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v9 )
  {
    v10 = (_QWORD *)v20;
    do
    {
      if ( v10 )
      {
        v12 = (_QWORD *)*v8;
        *v10 = *v8;
        v13 = (_QWORD *)v8[1];
        v10[1] = v13;
        v10[2] = v8[2];
        if ( v12 == v8 )
        {
          v10[1] = v10;
          v11 = v10;
          *v10 = v10;
        }
        else
        {
          *v13 = v10;
          *(_QWORD *)(*v10 + 8LL) = v10;
          v8[1] = v8;
          *v8 = v8;
          v8[2] = 0;
          v11 = (_QWORD *)*v10;
        }
        v10[3] = v11;
        v10[4] = v8[4];
        v10[5] = v8[5];
        v10[6] = v8[6];
        v8[6] = 0;
        v8[5] = 0;
        v8[4] = 0;
      }
      v8 += 7;
      v10 += 7;
    }
    while ( v9 != v8 );
    v14 = *(_QWORD **)a1;
    v9 = (_QWORD *)(*(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v9 )
    {
      do
      {
        v15 = *(v9 - 3);
        v9 -= 7;
        if ( v15 )
          j_j___libc_free_0(v15);
        v16 = (_QWORD *)*v9;
        while ( v9 != v16 )
        {
          v17 = (unsigned __int64)v16;
          v16 = (_QWORD *)*v16;
          j_j___libc_free_0(v17);
        }
      }
      while ( v9 != v14 );
      v9 = *(_QWORD **)a1;
    }
  }
  v18 = v21[0];
  if ( v6 != v9 )
    _libc_free((unsigned __int64)v9);
  *(_DWORD *)(a1 + 12) = v18;
  *(_QWORD *)a1 = v20;
  return v20;
}
