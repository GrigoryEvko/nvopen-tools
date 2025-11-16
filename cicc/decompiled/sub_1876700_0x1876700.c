// Function: sub_1876700
// Address: 0x1876700
//
__int64 __fastcall sub_1876700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // [rsp+8h] [rbp-48h]
  unsigned __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v5 = a1;
  v6 = a5;
  v7 = a3;
  if ( a4 != a3 && a2 != a1 )
  {
    v8 = a5 + 8;
    do
    {
      v9 = *(_QWORD *)(v8 + 8);
      if ( *(_QWORD *)(v7 + 40) >= *(_QWORD *)(v5 + 40) )
      {
        while ( v9 )
        {
          sub_1876060(*(_QWORD *)(v9 + 24));
          v26 = v9;
          v9 = *(_QWORD *)(v9 + 16);
          j_j___libc_free_0(v26, 40);
        }
        *(_QWORD *)(v8 + 8) = 0;
        *(_QWORD *)(v8 + 16) = v8;
        *(_QWORD *)(v8 + 24) = v8;
        *(_QWORD *)(v8 + 32) = 0;
        if ( *(_QWORD *)(v5 + 16) )
        {
          *(_DWORD *)v8 = *(_DWORD *)(v5 + 8);
          v27 = *(_QWORD *)(v5 + 16);
          *(_QWORD *)(v8 + 8) = v27;
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v5 + 24);
          *(_QWORD *)(v8 + 24) = *(_QWORD *)(v5 + 32);
          *(_QWORD *)(v27 + 8) = v8;
          *(_QWORD *)(v8 + 32) = *(_QWORD *)(v5 + 40);
          *(_QWORD *)(v5 + 16) = 0;
          *(_QWORD *)(v5 + 24) = v5 + 8;
          *(_QWORD *)(v5 + 32) = v5 + 8;
          *(_QWORD *)(v5 + 40) = 0;
        }
        v5 += 48;
      }
      else
      {
        while ( v9 )
        {
          sub_1876060(*(_QWORD *)(v9 + 24));
          v10 = v9;
          v9 = *(_QWORD *)(v9 + 16);
          j_j___libc_free_0(v10, 40);
        }
        *(_QWORD *)(v8 + 8) = 0;
        *(_QWORD *)(v8 + 16) = v8;
        *(_QWORD *)(v8 + 24) = v8;
        *(_QWORD *)(v8 + 32) = 0;
        if ( *(_QWORD *)(v7 + 16) )
        {
          *(_DWORD *)v8 = *(_DWORD *)(v7 + 8);
          v11 = *(_QWORD *)(v7 + 16);
          *(_QWORD *)(v8 + 8) = v11;
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 24);
          *(_QWORD *)(v8 + 24) = *(_QWORD *)(v7 + 32);
          *(_QWORD *)(v11 + 8) = v8;
          *(_QWORD *)(v8 + 32) = *(_QWORD *)(v7 + 40);
          *(_QWORD *)(v7 + 16) = 0;
          *(_QWORD *)(v7 + 24) = v7 + 8;
          *(_QWORD *)(v7 + 32) = v7 + 8;
          *(_QWORD *)(v7 + 40) = 0;
        }
        v7 += 48;
      }
      v6 = v8 + 40;
      v8 += 48;
    }
    while ( a2 != v5 && a4 != v7 );
  }
  v28 = a2 - v5;
  v30 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v5) >> 4);
  if ( a2 - v5 > 0 )
  {
    v12 = v6 + 8;
    v13 = v5 + 8;
    do
    {
      v14 = *(_QWORD *)(v12 + 8);
      while ( v14 )
      {
        sub_1876060(*(_QWORD *)(v14 + 24));
        v15 = v14;
        v14 = *(_QWORD *)(v14 + 16);
        j_j___libc_free_0(v15, 40);
      }
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = v12;
      *(_QWORD *)(v12 + 24) = v12;
      *(_QWORD *)(v12 + 32) = 0;
      if ( *(_QWORD *)(v13 + 8) )
      {
        *(_DWORD *)v12 = *(_DWORD *)v13;
        v16 = *(_QWORD *)(v13 + 8);
        *(_QWORD *)(v12 + 8) = v16;
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v13 + 16);
        *(_QWORD *)(v12 + 24) = *(_QWORD *)(v13 + 24);
        *(_QWORD *)(v16 + 8) = v12;
        *(_QWORD *)(v12 + 32) = *(_QWORD *)(v13 + 32);
        *(_QWORD *)(v13 + 8) = 0;
        *(_QWORD *)(v13 + 16) = v13;
        *(_QWORD *)(v13 + 24) = v13;
        *(_QWORD *)(v13 + 32) = 0;
      }
      v12 += 48;
      v13 += 48;
      --v30;
    }
    while ( v30 );
    v17 = v28;
    if ( v28 <= 0 )
      v17 = 48;
    v6 += v17;
  }
  v31 = a4 - v7;
  v18 = 0xAAAAAAAAAAAAAAABLL * ((a4 - v7) >> 4);
  if ( a4 - v7 > 0 )
  {
    v19 = v6 + 8;
    v20 = v7 + 8;
    do
    {
      v21 = *(_QWORD *)(v19 + 8);
      while ( v21 )
      {
        sub_1876060(*(_QWORD *)(v21 + 24));
        v22 = v21;
        v21 = *(_QWORD *)(v21 + 16);
        j_j___libc_free_0(v22, 40);
      }
      *(_QWORD *)(v19 + 8) = 0;
      *(_QWORD *)(v19 + 16) = v19;
      *(_QWORD *)(v19 + 24) = v19;
      *(_QWORD *)(v19 + 32) = 0;
      if ( *(_QWORD *)(v20 + 8) )
      {
        *(_DWORD *)v19 = *(_DWORD *)v20;
        v23 = *(_QWORD *)(v20 + 8);
        *(_QWORD *)(v19 + 8) = v23;
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v19 + 24) = *(_QWORD *)(v20 + 24);
        *(_QWORD *)(v23 + 8) = v19;
        *(_QWORD *)(v19 + 32) = *(_QWORD *)(v20 + 32);
        *(_QWORD *)(v20 + 8) = 0;
        *(_QWORD *)(v20 + 16) = v20;
        *(_QWORD *)(v20 + 24) = v20;
        *(_QWORD *)(v20 + 32) = 0;
      }
      v19 += 48;
      v20 += 48;
      --v18;
    }
    while ( v18 );
    v24 = v31;
    if ( v31 <= 0 )
      v24 = 48;
    v6 += v24;
  }
  return v6;
}
