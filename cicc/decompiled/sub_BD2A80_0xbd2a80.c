// Function: sub_BD2A80
// Address: 0xbd2a80
//
void __fastcall sub_BD2A80(__int64 a1, unsigned int a2, char a3)
{
  __int64 v3; // r15
  __int64 v5; // rbx
  __int64 v6; // r10
  __int64 v7; // r12
  const void *v8; // r13
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 *v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rsi
  size_t v15; // rbx
  __int64 v16; // [rsp+0h] [rbp-40h]

  v3 = a2;
  v5 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v6 = 32 * v5;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v7 = *(_QWORD *)(a1 - 8);
    v8 = (const void *)(v7 + v6);
  }
  else
  {
    v8 = (const void *)a1;
    v7 = a1 - v6;
  }
  v16 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  sub_BD2A10(a1, a2, a3);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v9 = *(_QWORD *)(a1 - 8);
  else
    v9 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v10 = v9;
  v11 = (__int64 *)v7;
  if ( v16 )
  {
    do
    {
      v12 = *v11;
      if ( *(_QWORD *)v10 )
      {
        v13 = *(_QWORD *)(v10 + 8);
        **(_QWORD **)(v10 + 16) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v10 + 16);
      }
      *(_QWORD *)v10 = v12;
      if ( v12 )
      {
        v14 = *(_QWORD *)(v12 + 16);
        *(_QWORD *)(v10 + 8) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = v10 + 8;
        *(_QWORD *)(v10 + 16) = v12 + 16;
        *(_QWORD *)(v12 + 16) = v10;
      }
      v10 += 32;
      v11 += 4;
    }
    while ( v10 != v9 + v16 );
  }
  if ( a3 )
  {
    v15 = 8 * v5;
    if ( v15 )
      memmove((void *)(v9 + 32 * v3), v8, v15);
  }
  sub_BD2950(v7, (__int64)v8, 1);
}
