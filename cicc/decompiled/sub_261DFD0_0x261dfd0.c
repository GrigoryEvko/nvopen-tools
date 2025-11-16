// Function: sub_261DFD0
// Address: 0x261dfd0
//
__int64 __fastcall sub_261DFD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r15
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // r14
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rdi
  __int64 v17; // rdx
  unsigned __int64 v19; // rdi
  __int64 v20; // rsi
  unsigned __int64 v21; // r14
  __int64 v22; // r12
  __int64 v23; // r15
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v29; // [rsp+8h] [rbp-38h]
  __int64 v30; // [rsp+8h] [rbp-38h]

  if ( a1 == a2 )
  {
LABEL_11:
    v12 = a4 - a3;
    v29 = 0xAAAAAAAAAAAAAAABLL * ((a4 - a3) >> 4);
    if ( v12 > 0 )
    {
      v13 = a5 + 8;
      v14 = a3 + 8;
      do
      {
        v15 = *(_QWORD *)(v13 + 8);
        while ( v15 )
        {
          sub_261DCB0(*(_QWORD *)(v15 + 24));
          v16 = v15;
          v15 = *(_QWORD *)(v15 + 16);
          j_j___libc_free_0(v16);
        }
        *(_QWORD *)(v13 + 8) = 0;
        *(_QWORD *)(v13 + 16) = v13;
        *(_QWORD *)(v13 + 24) = v13;
        *(_QWORD *)(v13 + 32) = 0;
        if ( *(_QWORD *)(v14 + 8) )
        {
          *(_DWORD *)v13 = *(_DWORD *)v14;
          v17 = *(_QWORD *)(v14 + 8);
          *(_QWORD *)(v13 + 8) = v17;
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v14 + 16);
          *(_QWORD *)(v13 + 24) = *(_QWORD *)(v14 + 24);
          *(_QWORD *)(v17 + 8) = v13;
          *(_QWORD *)(v13 + 32) = *(_QWORD *)(v14 + 32);
          *(_QWORD *)(v14 + 8) = 0;
          *(_QWORD *)(v14 + 16) = v14;
          *(_QWORD *)(v14 + 24) = v14;
          *(_QWORD *)(v14 + 32) = 0;
        }
        v13 += 48;
        v14 += 48;
        --v29;
      }
      while ( v29 );
      return a5 + v12;
    }
    return a5;
  }
  v8 = a1;
  while ( a4 != a3 )
  {
    v9 = *(_QWORD *)(a5 + 16);
    if ( *(_QWORD *)(a3 + 40) >= *(_QWORD *)(v8 + 40) )
    {
      while ( v9 )
      {
        sub_261DCB0(*(_QWORD *)(v9 + 24));
        v19 = v9;
        v9 = *(_QWORD *)(v9 + 16);
        j_j___libc_free_0(v19);
      }
      *(_QWORD *)(a5 + 16) = 0;
      *(_QWORD *)(a5 + 24) = a5 + 8;
      *(_QWORD *)(a5 + 32) = a5 + 8;
      *(_QWORD *)(a5 + 40) = 0;
      if ( *(_QWORD *)(v8 + 16) )
      {
        *(_DWORD *)(a5 + 8) = *(_DWORD *)(v8 + 8);
        v20 = *(_QWORD *)(v8 + 16);
        *(_QWORD *)(a5 + 16) = v20;
        *(_QWORD *)(a5 + 24) = *(_QWORD *)(v8 + 24);
        *(_QWORD *)(a5 + 32) = *(_QWORD *)(v8 + 32);
        *(_QWORD *)(v20 + 8) = a5 + 8;
        *(_QWORD *)(a5 + 40) = *(_QWORD *)(v8 + 40);
        *(_QWORD *)(v8 + 16) = 0;
        *(_QWORD *)(v8 + 24) = v8 + 8;
        *(_QWORD *)(v8 + 32) = v8 + 8;
        *(_QWORD *)(v8 + 40) = 0;
      }
      v8 += 48;
    }
    else
    {
      while ( v9 )
      {
        sub_261DCB0(*(_QWORD *)(v9 + 24));
        v10 = v9;
        v9 = *(_QWORD *)(v9 + 16);
        j_j___libc_free_0(v10);
      }
      *(_QWORD *)(a5 + 16) = 0;
      *(_QWORD *)(a5 + 24) = a5 + 8;
      *(_QWORD *)(a5 + 32) = a5 + 8;
      *(_QWORD *)(a5 + 40) = 0;
      if ( *(_QWORD *)(a3 + 16) )
      {
        *(_DWORD *)(a5 + 8) = *(_DWORD *)(a3 + 8);
        v11 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(a5 + 16) = v11;
        *(_QWORD *)(a5 + 24) = *(_QWORD *)(a3 + 24);
        *(_QWORD *)(a5 + 32) = *(_QWORD *)(a3 + 32);
        *(_QWORD *)(v11 + 8) = a5 + 8;
        *(_QWORD *)(a5 + 40) = *(_QWORD *)(a3 + 40);
        *(_QWORD *)(a3 + 16) = 0;
        *(_QWORD *)(a3 + 24) = a3 + 8;
        *(_QWORD *)(a3 + 32) = a3 + 8;
        *(_QWORD *)(a3 + 40) = 0;
      }
      a3 += 48;
    }
    a5 += 48;
    if ( v8 == a2 )
      goto LABEL_11;
  }
  v30 = a2 - v8;
  v21 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v8) >> 4);
  if ( a2 - v8 <= 0 )
    return a5;
  v22 = a5 + 8;
  v23 = v8 + 8;
  do
  {
    v24 = *(_QWORD *)(v22 + 8);
    while ( v24 )
    {
      sub_261DCB0(*(_QWORD *)(v24 + 24));
      v25 = v24;
      v24 = *(_QWORD *)(v24 + 16);
      j_j___libc_free_0(v25);
    }
    *(_QWORD *)(v22 + 8) = 0;
    *(_QWORD *)(v22 + 16) = v22;
    *(_QWORD *)(v22 + 24) = v22;
    *(_QWORD *)(v22 + 32) = 0;
    if ( *(_QWORD *)(v23 + 8) )
    {
      *(_DWORD *)v22 = *(_DWORD *)v23;
      v26 = *(_QWORD *)(v23 + 8);
      *(_QWORD *)(v22 + 8) = v26;
      *(_QWORD *)(v22 + 16) = *(_QWORD *)(v23 + 16);
      *(_QWORD *)(v22 + 24) = *(_QWORD *)(v23 + 24);
      *(_QWORD *)(v26 + 8) = v22;
      *(_QWORD *)(v22 + 32) = *(_QWORD *)(v23 + 32);
      *(_QWORD *)(v23 + 8) = 0;
      *(_QWORD *)(v23 + 16) = v23;
      *(_QWORD *)(v23 + 24) = v23;
      *(_QWORD *)(v23 + 32) = 0;
    }
    v22 += 48;
    v23 += 48;
    --v21;
  }
  while ( v21 );
  v27 = 48;
  if ( v30 > 0 )
    v27 = v30;
  return a5 + v27;
}
