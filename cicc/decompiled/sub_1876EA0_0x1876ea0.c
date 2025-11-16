// Function: sub_1876EA0
// Address: 0x1876ea0
//
__int64 __fastcall sub_1876EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // r12
  __int64 v21; // r13
  __int64 v22; // r15
  __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-48h]
  unsigned __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v5 = a1;
  v6 = a5;
  v7 = a3;
  if ( a4 != a3 && a2 != a1 )
  {
    v8 = a5 + 8;
    do
    {
      v9 = *(_QWORD *)(v8 + 8);
      if ( *(_QWORD *)(v7 + 48) <= *(_QWORD *)(v5 + 48) )
      {
        while ( v9 )
        {
          sub_1876060(*(_QWORD *)(v9 + 24));
          v29 = v9;
          v9 = *(_QWORD *)(v9 + 16);
          j_j___libc_free_0(v29, 40);
        }
        *(_QWORD *)(v8 + 8) = 0;
        *(_QWORD *)(v8 + 16) = v8;
        *(_QWORD *)(v8 + 24) = v8;
        *(_QWORD *)(v8 + 32) = 0;
        if ( *(_QWORD *)(v5 + 16) )
        {
          *(_DWORD *)v8 = *(_DWORD *)(v5 + 8);
          v30 = *(_QWORD *)(v5 + 16);
          *(_QWORD *)(v8 + 8) = v30;
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v5 + 24);
          *(_QWORD *)(v8 + 24) = *(_QWORD *)(v5 + 32);
          *(_QWORD *)(v30 + 8) = v8;
          *(_QWORD *)(v8 + 32) = *(_QWORD *)(v5 + 40);
          *(_QWORD *)(v5 + 16) = 0;
          *(_QWORD *)(v5 + 24) = v5 + 8;
          *(_QWORD *)(v5 + 32) = v5 + 8;
          *(_QWORD *)(v5 + 40) = 0;
        }
        v31 = *(_QWORD *)(v5 + 48);
        v5 += 80;
        *(_QWORD *)(v8 + 40) = v31;
        *(_QWORD *)(v8 + 48) = *(_QWORD *)(v5 - 24);
        *(_QWORD *)(v8 + 56) = *(_QWORD *)(v5 - 16);
        *(_QWORD *)(v8 + 64) = *(_QWORD *)(v5 - 8);
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
        v12 = *(_QWORD *)(v7 + 48);
        v7 += 80;
        *(_QWORD *)(v8 + 40) = v12;
        *(_QWORD *)(v8 + 48) = *(_QWORD *)(v7 - 24);
        *(_QWORD *)(v8 + 56) = *(_QWORD *)(v7 - 16);
        *(_QWORD *)(v8 + 64) = *(_QWORD *)(v7 - 8);
      }
      v6 = v8 + 72;
      v8 += 80;
    }
    while ( a2 != v5 && a4 != v7 );
  }
  v32 = a2 - v5;
  v34 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - v5) >> 4);
  if ( a2 - v5 > 0 )
  {
    v13 = v6 + 8;
    v14 = v5 + 8;
    do
    {
      v15 = *(_QWORD *)(v13 + 8);
      while ( v15 )
      {
        sub_1876060(*(_QWORD *)(v15 + 24));
        v16 = v15;
        v15 = *(_QWORD *)(v15 + 16);
        j_j___libc_free_0(v16, 40);
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
      v18 = *(_QWORD *)(v14 + 40);
      v13 += 80;
      v14 += 80;
      *(_QWORD *)(v13 - 40) = v18;
      *(_QWORD *)(v13 - 32) = *(_QWORD *)(v14 - 32);
      *(_QWORD *)(v13 - 24) = *(_QWORD *)(v14 - 24);
      *(_QWORD *)(v13 - 16) = *(_QWORD *)(v14 - 16);
      --v34;
    }
    while ( v34 );
    v19 = v32;
    if ( v32 <= 0 )
      v19 = 80;
    v6 += v19;
  }
  v35 = a4 - v7;
  v20 = 0xCCCCCCCCCCCCCCCDLL * ((a4 - v7) >> 4);
  if ( a4 - v7 > 0 )
  {
    v21 = v6 + 8;
    v22 = v7 + 8;
    do
    {
      v23 = *(_QWORD *)(v21 + 8);
      while ( v23 )
      {
        sub_1876060(*(_QWORD *)(v23 + 24));
        v24 = v23;
        v23 = *(_QWORD *)(v23 + 16);
        j_j___libc_free_0(v24, 40);
      }
      *(_QWORD *)(v21 + 8) = 0;
      *(_QWORD *)(v21 + 16) = v21;
      *(_QWORD *)(v21 + 24) = v21;
      *(_QWORD *)(v21 + 32) = 0;
      if ( *(_QWORD *)(v22 + 8) )
      {
        *(_DWORD *)v21 = *(_DWORD *)v22;
        v25 = *(_QWORD *)(v22 + 8);
        *(_QWORD *)(v21 + 8) = v25;
        *(_QWORD *)(v21 + 16) = *(_QWORD *)(v22 + 16);
        *(_QWORD *)(v21 + 24) = *(_QWORD *)(v22 + 24);
        *(_QWORD *)(v25 + 8) = v21;
        *(_QWORD *)(v21 + 32) = *(_QWORD *)(v22 + 32);
        *(_QWORD *)(v22 + 8) = 0;
        *(_QWORD *)(v22 + 16) = v22;
        *(_QWORD *)(v22 + 24) = v22;
        *(_QWORD *)(v22 + 32) = 0;
      }
      v26 = *(_QWORD *)(v22 + 40);
      v21 += 80;
      v22 += 80;
      *(_QWORD *)(v21 - 40) = v26;
      *(_QWORD *)(v21 - 32) = *(_QWORD *)(v22 - 32);
      *(_QWORD *)(v21 - 24) = *(_QWORD *)(v22 - 24);
      *(_QWORD *)(v21 - 16) = *(_QWORD *)(v22 - 16);
      --v20;
    }
    while ( v20 );
    v27 = v35;
    if ( v35 <= 0 )
      v27 = 80;
    v6 += v27;
  }
  return v6;
}
