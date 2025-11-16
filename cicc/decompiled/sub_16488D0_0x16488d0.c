// Function: sub_16488D0
// Address: 0x16488d0
//
void __fastcall sub_16488D0(__int64 a1, unsigned int a2, char a3)
{
  __int64 v3; // r15
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 *v7; // r13
  _QWORD *v8; // r9
  _QWORD *v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 v14; // rsi
  size_t v15; // r12
  __int64 *v16; // [rsp+8h] [rbp-38h]

  v3 = a2;
  v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v6 = 3 * v5;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v7 = *(__int64 **)(a1 - 8);
    v16 = &v7[v6];
  }
  else
  {
    v16 = (__int64 *)a1;
    v7 = (__int64 *)(a1 - v6 * 8);
  }
  sub_1648880(a1, a2, a3);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v8 = *(_QWORD **)(a1 - 8);
  else
    v8 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v9 = v8;
  v10 = v7;
  if ( v6 * 8 )
  {
    do
    {
      v11 = *v10;
      if ( *v9 )
      {
        v12 = v9[1];
        v13 = v9[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
      }
      *v9 = v11;
      if ( v11 )
      {
        v14 = *(_QWORD *)(v11 + 8);
        v9[1] = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (unsigned __int64)(v9 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
        v9[2] = (v11 + 8) | v9[2] & 3LL;
        *(_QWORD *)(v11 + 8) = v9;
      }
      v9 += 3;
      v10 += 3;
    }
    while ( v9 != &v8[v6] );
  }
  if ( a3 )
  {
    v15 = 8 * v5;
    if ( v15 )
      memmove(&v8[3 * v3 + 1], &v7[v6 + 1], v15);
  }
  sub_1648630(v7, v16, 1);
}
