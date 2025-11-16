// Function: sub_2295D00
// Address: 0x2295d00
//
void __fastcall sub_2295D00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rcx
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // r8
  int v16; // r15d
  unsigned __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v20, a6);
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v7;
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v7 + 8);
        *(_DWORD *)(v9 + 16) = *(_DWORD *)(v7 + 16);
        *(_QWORD *)(v9 + 24) = *(_QWORD *)(v7 + 24);
        v10 = *(_QWORD *)(v7 + 32);
        *(_QWORD *)(v7 + 24) = 1;
        *(_QWORD *)(v9 + 32) = v10;
        v11 = *(_QWORD *)(v7 + 40);
        *(_QWORD *)(v7 + 32) = 1;
        *(_QWORD *)(v9 + 40) = v11;
        *(_QWORD *)(v7 + 40) = 1;
      }
      v7 += 48LL;
      v9 += 48;
    }
    while ( v8 != v7 );
    v12 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v13 = *(_QWORD *)(v8 - 8);
        v8 -= 48LL;
        if ( (v13 & 1) == 0 && v13 )
        {
          if ( *(_QWORD *)v13 != v13 + 16 )
          {
            v17 = v13;
            _libc_free(*(_QWORD *)v13);
            v13 = v17;
          }
          j_j___libc_free_0(v13);
        }
        v14 = *(_QWORD *)(v8 + 32);
        if ( (v14 & 1) == 0 && v14 )
        {
          if ( *(_QWORD *)v14 != v14 + 16 )
          {
            v18 = *(_QWORD *)(v8 + 32);
            _libc_free(*(_QWORD *)v14);
            v14 = v18;
          }
          j_j___libc_free_0(v14);
        }
        v15 = *(_QWORD *)(v8 + 24);
        if ( (v15 & 1) == 0 && v15 )
        {
          if ( *(_QWORD *)v15 != v15 + 16 )
          {
            v19 = *(_QWORD *)(v8 + 24);
            _libc_free(*(_QWORD *)v15);
            v15 = v19;
          }
          j_j___libc_free_0(v15);
        }
      }
      while ( v12 != v8 );
      v8 = *(_QWORD *)a1;
    }
  }
  v16 = v20[0];
  if ( a1 + 16 != v8 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v16;
}
