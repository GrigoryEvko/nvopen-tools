// Function: sub_315E290
// Address: 0x315e290
//
void __fastcall sub_315E290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  int v8; // eax
  _QWORD *v9; // rdi
  unsigned __int64 *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r13
  __int64 v16; // r12
  _QWORD *v17; // rdi
  unsigned __int64 *v18; // rax
  unsigned __int64 *v19; // r12
  __int64 v20; // r15
  __int64 v21; // rdx
  unsigned __int64 *v22; // r15
  int v23; // r15d
  unsigned __int64 *v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v7 )
  {
    v10 = (unsigned __int64 *)(a1 + 16);
    v15 = sub_C8D7D0(a1, a1 + 16, 0, 0x30u, v25, a6);
    v16 = 48LL * *(unsigned int *)(a1 + 8);
    v17 = (_QWORD *)(v16 + v15);
    if ( v16 + v15 )
    {
      *v17 = v17 + 2;
      v17[1] = 0x400000000LL;
      v12 = *(unsigned int *)(a2 + 8);
      if ( (_DWORD)v12 )
        sub_315E080((__int64)v17, (char **)a2, v11, v12, v13, v14);
      v16 = 48LL * *(unsigned int *)(a1 + 8);
    }
    v18 = *(unsigned __int64 **)a1;
    v19 = (unsigned __int64 *)(*(_QWORD *)a1 + v16);
    if ( *(unsigned __int64 **)a1 != v19 )
    {
      v20 = v15;
      do
      {
        if ( v20 )
        {
          *(_DWORD *)(v20 + 8) = 0;
          *(_QWORD *)v20 = v20 + 16;
          *(_DWORD *)(v20 + 12) = 4;
          v21 = *((unsigned int *)v18 + 2);
          if ( (_DWORD)v21 )
          {
            v24 = v18;
            sub_315E080(v20, (char **)v18, v21, v12, v13, v14);
            v18 = v24;
          }
        }
        v18 += 6;
        v20 += 48;
      }
      while ( v19 != v18 );
      v22 = *(unsigned __int64 **)a1;
      v19 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
      if ( *(unsigned __int64 **)a1 != v19 )
      {
        do
        {
          v19 -= 6;
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
            _libc_free(*v19);
        }
        while ( v19 != v22 );
        v19 = *(unsigned __int64 **)a1;
      }
    }
    v23 = v25[0];
    if ( v10 != v19 )
      _libc_free((unsigned __int64)v19);
    *(_QWORD *)a1 = v15;
    *(_DWORD *)(a1 + 12) = v23;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 8);
    v9 = (_QWORD *)(*(_QWORD *)a1 + 48 * v7);
    if ( v9 )
    {
      *v9 = v9 + 2;
      v9[1] = 0x400000000LL;
      if ( *(_DWORD *)(a2 + 8) )
        sub_315E080((__int64)v9, (char **)a2, v7, a4, a5, a6);
      v8 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + 1;
  }
}
