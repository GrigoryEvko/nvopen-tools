// Function: sub_1880A60
// Address: 0x1880a60
//
void __fastcall sub_1880A60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rax
  int v12; // edi
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // r9d
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 i; // r12
  __int64 v21; // rdi
  __int64 v22; // rcx
  int v23; // esi
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v3 = 0x2AAAAAAAAAAAAAALL;
  *a1 = a3;
  a1[1] = 0;
  if ( a3 <= 0x2AAAAAAAAAAAAAALL )
    v3 = a3;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v6 = 48 * v3;
      v7 = sub_2207800(48 * v3, &unk_435FF63);
      v8 = v7;
      if ( v7 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v9 = v7 + v6;
    v10 = v7 + 8;
    v26 = a2 + 8;
    v11 = *(_QWORD *)(a2 + 16);
    if ( v11 )
    {
      v12 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)(v8 + 16) = v11;
      *(_DWORD *)(v8 + 8) = v12;
      *(_QWORD *)(v8 + 24) = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(v8 + 32) = *(_QWORD *)(a2 + 32);
      *(_QWORD *)(v11 + 8) = v10;
      v13 = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(a2 + 16) = 0;
      *(_QWORD *)(v8 + 40) = v13;
      *(_QWORD *)(a2 + 40) = 0;
      *(_QWORD *)(a2 + 24) = v26;
      *(_QWORD *)(a2 + 32) = v26;
    }
    else
    {
      *(_DWORD *)(v8 + 8) = 0;
      *(_QWORD *)(v8 + 16) = 0;
      *(_QWORD *)(v8 + 24) = v10;
      *(_QWORD *)(v8 + 32) = v10;
      *(_QWORD *)(v8 + 40) = 0;
    }
    v14 = v8 + 48;
    if ( v9 == v8 + 48 )
    {
      v25 = v8;
    }
    else
    {
      do
      {
        while ( 1 )
        {
          v17 = *(_QWORD *)(v14 - 32);
          v18 = v14 - 40;
          v19 = v14 + 8;
          if ( !v17 )
            break;
          v15 = *(_DWORD *)(v14 - 40);
          *(_QWORD *)(v14 + 16) = v17;
          v14 += 48;
          *(_DWORD *)(v14 - 40) = v15;
          *(_QWORD *)(v14 - 24) = *(_QWORD *)(v14 - 72);
          *(_QWORD *)(v14 - 16) = *(_QWORD *)(v14 - 64);
          *(_QWORD *)(v17 + 8) = v19;
          v16 = *(_QWORD *)(v14 - 56);
          *(_QWORD *)(v14 - 80) = 0;
          *(_QWORD *)(v14 - 8) = v16;
          *(_QWORD *)(v14 - 72) = v18;
          *(_QWORD *)(v14 - 64) = v18;
          *(_QWORD *)(v14 - 56) = 0;
          if ( v9 == v14 )
            goto LABEL_12;
        }
        *(_DWORD *)(v14 + 8) = 0;
        v14 += 48;
        *(_QWORD *)(v14 - 32) = 0;
        *(_QWORD *)(v14 - 24) = v19;
        *(_QWORD *)(v14 - 16) = v19;
        *(_QWORD *)(v14 - 8) = 0;
      }
      while ( v9 != v14 );
LABEL_12:
      v25 = v8 + 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v6 - 96) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3);
    }
    for ( i = *(_QWORD *)(a2 + 16); i; v25 = v27 )
    {
      v27 = v25;
      sub_1876060(*(_QWORD *)(i + 24));
      v21 = i;
      i = *(_QWORD *)(i + 16);
      j_j___libc_free_0(v21, 40);
    }
    v22 = *(_QWORD *)(v25 + 16);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 40) = 0;
    *(_QWORD *)(a2 + 24) = v26;
    *(_QWORD *)(a2 + 32) = v26;
    if ( v22 )
    {
      v23 = *(_DWORD *)(v25 + 8);
      *(_QWORD *)(a2 + 16) = v22;
      *(_DWORD *)(a2 + 8) = v23;
      *(_QWORD *)(a2 + 24) = *(_QWORD *)(v25 + 24);
      *(_QWORD *)(a2 + 32) = *(_QWORD *)(v25 + 32);
      *(_QWORD *)(v22 + 8) = v26;
      v24 = *(_QWORD *)(v25 + 40);
      *(_QWORD *)(v25 + 16) = 0;
      *(_QWORD *)(a2 + 40) = v24;
      *(_QWORD *)(v25 + 24) = v25 + 8;
      *(_QWORD *)(v25 + 32) = v25 + 8;
      *(_QWORD *)(v25 + 40) = 0;
    }
    a1[2] = v8;
    a1[1] = v3;
  }
}
