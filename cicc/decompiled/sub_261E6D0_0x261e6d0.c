// Function: sub_261E6D0
// Address: 0x261e6d0
//
__int64 __fastcall sub_261E6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r15
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r12
  __int64 v15; // r14
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdx
  unsigned __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  __int64 v25; // r12
  __int64 v26; // r15
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]

  if ( a1 == a2 )
  {
LABEL_11:
    v13 = a4 - a3;
    v33 = 0xCCCCCCCCCCCCCCCDLL * ((a4 - a3) >> 4);
    if ( v13 > 0 )
    {
      v14 = a5 + 8;
      v15 = a3 + 8;
      do
      {
        v16 = *(_QWORD *)(v14 + 8);
        while ( v16 )
        {
          sub_261DCB0(*(_QWORD *)(v16 + 24));
          v17 = v16;
          v16 = *(_QWORD *)(v16 + 16);
          j_j___libc_free_0(v17);
        }
        *(_QWORD *)(v14 + 8) = 0;
        *(_QWORD *)(v14 + 16) = v14;
        *(_QWORD *)(v14 + 24) = v14;
        *(_QWORD *)(v14 + 32) = 0;
        if ( *(_QWORD *)(v15 + 8) )
        {
          *(_DWORD *)v14 = *(_DWORD *)v15;
          v18 = *(_QWORD *)(v15 + 8);
          *(_QWORD *)(v14 + 8) = v18;
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v15 + 16);
          *(_QWORD *)(v14 + 24) = *(_QWORD *)(v15 + 24);
          *(_QWORD *)(v18 + 8) = v14;
          *(_QWORD *)(v14 + 32) = *(_QWORD *)(v15 + 32);
          *(_QWORD *)(v15 + 8) = 0;
          *(_QWORD *)(v15 + 16) = v15;
          *(_QWORD *)(v15 + 24) = v15;
          *(_QWORD *)(v15 + 32) = 0;
        }
        v19 = *(_QWORD *)(v15 + 40);
        v14 += 80;
        v15 += 80;
        *(_QWORD *)(v14 - 40) = v19;
        *(_QWORD *)(v14 - 32) = *(_QWORD *)(v15 - 32);
        *(_QWORD *)(v14 - 24) = *(_QWORD *)(v15 - 24);
        *(_QWORD *)(v14 - 16) = *(_QWORD *)(v15 - 16);
        --v33;
      }
      while ( v33 );
      return a5 + v13;
    }
    return a5;
  }
  v8 = a1;
  while ( a4 != a3 )
  {
    v9 = *(_QWORD *)(a5 + 16);
    if ( *(_QWORD *)(a3 + 48) <= *(_QWORD *)(v8 + 48) )
    {
      while ( v9 )
      {
        sub_261DCB0(*(_QWORD *)(v9 + 24));
        v21 = v9;
        v9 = *(_QWORD *)(v9 + 16);
        j_j___libc_free_0(v21);
      }
      *(_QWORD *)(a5 + 16) = 0;
      *(_QWORD *)(a5 + 24) = a5 + 8;
      *(_QWORD *)(a5 + 32) = a5 + 8;
      *(_QWORD *)(a5 + 40) = 0;
      if ( *(_QWORD *)(v8 + 16) )
      {
        *(_DWORD *)(a5 + 8) = *(_DWORD *)(v8 + 8);
        v22 = *(_QWORD *)(v8 + 16);
        *(_QWORD *)(a5 + 16) = v22;
        *(_QWORD *)(a5 + 24) = *(_QWORD *)(v8 + 24);
        *(_QWORD *)(a5 + 32) = *(_QWORD *)(v8 + 32);
        *(_QWORD *)(v22 + 8) = a5 + 8;
        *(_QWORD *)(a5 + 40) = *(_QWORD *)(v8 + 40);
        *(_QWORD *)(v8 + 16) = 0;
        *(_QWORD *)(v8 + 24) = v8 + 8;
        *(_QWORD *)(v8 + 32) = v8 + 8;
        *(_QWORD *)(v8 + 40) = 0;
      }
      v23 = *(_QWORD *)(v8 + 48);
      v8 += 80;
      *(_QWORD *)(a5 + 48) = v23;
      *(_QWORD *)(a5 + 56) = *(_QWORD *)(v8 - 24);
      *(_QWORD *)(a5 + 64) = *(_QWORD *)(v8 - 16);
      *(_QWORD *)(a5 + 72) = *(_QWORD *)(v8 - 8);
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
      v12 = *(_QWORD *)(a3 + 48);
      a3 += 80;
      *(_QWORD *)(a5 + 48) = v12;
      *(_QWORD *)(a5 + 56) = *(_QWORD *)(a3 - 24);
      *(_QWORD *)(a5 + 64) = *(_QWORD *)(a3 - 16);
      *(_QWORD *)(a5 + 72) = *(_QWORD *)(a3 - 8);
    }
    a5 += 80;
    if ( v8 == a2 )
      goto LABEL_11;
  }
  v34 = a2 - v8;
  v24 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - v8) >> 4);
  if ( a2 - v8 <= 0 )
    return a5;
  v25 = a5 + 8;
  v26 = v8 + 8;
  do
  {
    v27 = *(_QWORD *)(v25 + 8);
    while ( v27 )
    {
      sub_261DCB0(*(_QWORD *)(v27 + 24));
      v28 = v27;
      v27 = *(_QWORD *)(v27 + 16);
      j_j___libc_free_0(v28);
    }
    *(_QWORD *)(v25 + 8) = 0;
    *(_QWORD *)(v25 + 16) = v25;
    *(_QWORD *)(v25 + 24) = v25;
    *(_QWORD *)(v25 + 32) = 0;
    if ( *(_QWORD *)(v26 + 8) )
    {
      *(_DWORD *)v25 = *(_DWORD *)v26;
      v29 = *(_QWORD *)(v26 + 8);
      *(_QWORD *)(v25 + 8) = v29;
      *(_QWORD *)(v25 + 16) = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v25 + 24) = *(_QWORD *)(v26 + 24);
      *(_QWORD *)(v29 + 8) = v25;
      *(_QWORD *)(v25 + 32) = *(_QWORD *)(v26 + 32);
      *(_QWORD *)(v26 + 8) = 0;
      *(_QWORD *)(v26 + 16) = v26;
      *(_QWORD *)(v26 + 24) = v26;
      *(_QWORD *)(v26 + 32) = 0;
    }
    v30 = *(_QWORD *)(v26 + 40);
    v25 += 80;
    v26 += 80;
    *(_QWORD *)(v25 - 40) = v30;
    *(_QWORD *)(v25 - 32) = *(_QWORD *)(v26 - 32);
    *(_QWORD *)(v25 - 24) = *(_QWORD *)(v26 - 24);
    *(_QWORD *)(v25 - 16) = *(_QWORD *)(v26 - 16);
    --v24;
  }
  while ( v24 );
  v31 = 80;
  if ( v34 > 0 )
    v31 = v34;
  return a5 + v31;
}
