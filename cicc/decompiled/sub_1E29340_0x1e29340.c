// Function: sub_1E29340
// Address: 0x1e29340
//
__int64 __fastcall sub_1E29340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  const void *v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rcx
  _BYTE *i; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // r9
  __int64 v17; // r14
  __int64 v18; // rbx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  const void *v22; // [rsp+0h] [rbp-90h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  __int64 v24; // [rsp+28h] [rbp-68h]
  _BYTE *v25; // [rsp+30h] [rbp-60h] BYREF
  __int64 v26; // [rsp+38h] [rbp-58h]
  _BYTE v27[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = v6;
  v22 = v6;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v7 = *(_QWORD *)(a2 + 40);
  v25 = v27;
  v8 = *(_QWORD *)(a2 + 32);
  v26 = 0x400000000LL;
  if ( v8 != v7 )
  {
    v9 = *(_QWORD *)(v7 - 8);
    v10 = 0;
    v23 = v7 - 8;
    for ( i = v27; ; i = v25 )
    {
      *(_QWORD *)&i[8 * v10] = v9;
      LODWORD(v10) = v26 + 1;
      LODWORD(v26) = v26 + 1;
      do
      {
        v12 = (unsigned __int64)v25;
        v13 = (unsigned int)(v10 - 1);
        v14 = *(_QWORD *)&v25[8 * (unsigned int)v10 - 8];
        LODWORD(v26) = v10 - 1;
        v15 = *(_QWORD *)(v14 + 16);
        v16 = v15 - *(_QWORD *)(v14 + 8);
        v17 = v16 >> 3;
        v18 = v16 >> 3;
        if ( v16 >> 3 > (unsigned __int64)HIDWORD(v26) - v13 )
        {
          v24 = *(_QWORD *)(v14 + 16) - *(_QWORD *)(v14 + 8);
          sub_16CD150((__int64)&v25, v27, v17 + v13, 8, a5, v16);
          v12 = (unsigned __int64)v25;
          v13 = (unsigned int)v26;
          v16 = v24;
        }
        v19 = v12 + 8 * v13;
        if ( v16 > 0 )
        {
          do
          {
            v19 += 8LL;
            *(_QWORD *)(v19 - 8) = *(_QWORD *)(v15 - 8 * v17 + 8 * v18-- - 8);
          }
          while ( v18 );
          LODWORD(v13) = v26;
        }
        v20 = *(unsigned int *)(a1 + 8);
        LODWORD(v26) = v17 + v13;
        if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, v22, 0, 8, a5, v16);
          v20 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v20) = v14;
        v10 = (unsigned int)v26;
        ++*(_DWORD *)(a1 + 8);
      }
      while ( (_DWORD)v10 );
      if ( v8 == v23 )
        break;
      v9 = *(_QWORD *)(v23 - 8);
      if ( !HIDWORD(v26) )
      {
        sub_16CD150((__int64)&v25, v27, 0, 8, a5, v16);
        v10 = (unsigned int)v26;
      }
      v23 -= 8;
    }
    if ( v25 != v27 )
      _libc_free((unsigned __int64)v25);
  }
  return a1;
}
