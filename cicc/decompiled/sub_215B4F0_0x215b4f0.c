// Function: sub_215B4F0
// Address: 0x215b4f0
//
__int64 __fastcall sub_215B4F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  int v7; // esi
  int v8; // r14d
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rdi
  _QWORD *i; // rax
  _QWORD *v17; // rsi
  unsigned __int64 *v18; // rdi
  unsigned __int64 v19; // rsi
  __int64 *v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r9
  __int64 v23; // rsi
  unsigned __int8 v25; // [rsp+Eh] [rbp-42h]
  char v26; // [rsp+Fh] [rbp-41h]
  __int64 *v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 272);
  v26 = 0;
  if ( v4 )
    v26 = *(_BYTE *)(v4 + 1744);
  if ( !*(_BYTE *)(a1 + 784) )
  {
    sub_215A100(a1, a2);
    *(_BYTE *)(a1 + 784) = 1;
  }
  v5 = a2 + 8;
  sub_2154C60(a1, a2);
  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 == a2 + 8 )
  {
    v9 = 0;
    v8 = 0;
  }
  else
  {
    v7 = 0;
    do
    {
      v6 = *(_QWORD *)(v6 + 8);
      ++v7;
    }
    while ( v6 != v5 );
    v8 = v7;
    v9 = 8LL * v7;
    if ( (unsigned __int64)v7 > 0xFFFFFFFFFFFFFFFLL )
      v9 = -1;
  }
  v10 = sub_2207820(v9);
  v13 = *(_QWORD **)(a2 + 16);
  v14 = 0;
  v27 = (__int64 *)v10;
  v15 = v10;
  for ( i = v13; (_QWORD *)v5 != i; i = (_QWORD *)i[1] )
  {
    v17 = i - 7;
    if ( !i )
      v17 = 0;
    v15 += 8;
    *(_QWORD *)(v15 - 8) = v17;
  }
  if ( v5 != (*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    while ( 1 )
    {
      sub_1631C10(a2 + 8, (__int64)(v13 - 7));
      v18 = (unsigned __int64 *)v13[1];
      v19 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
      *v18 = v19 | *v18 & 7;
      *(_QWORD *)(v19 + 8) = v18;
      v13[1] = 0;
      *v13 &= 7uLL;
      if ( v5 == (*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
        break;
      v13 = *(_QWORD **)(a2 + 16);
    }
  }
  v25 = sub_3972F10(a1, a2, v11, v12, v14);
  if ( v8 > 0 )
  {
    v20 = v27;
    v28 = (__int64)&v27[(unsigned int)(v8 - 1) + 1];
    do
    {
      v21 = *v20++;
      sub_1631BE0(a2 + 8, v21);
      v22 = *(_QWORD *)(a2 + 8);
      v23 = *(_QWORD *)(v21 + 56);
      *(_QWORD *)(v21 + 64) = v5;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v21 + 56) = v22 | v23 & 7;
      *(_QWORD *)(v22 + 8) = v21 + 56;
      *(_QWORD *)(a2 + 8) = *(_QWORD *)(a2 + 8) & 7LL | (v21 + 56);
    }
    while ( (__int64 *)v28 != v20 );
  }
  if ( v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 )
    sub_214A860(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 16LL));
  sub_214A560(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 16LL));
  return v25;
}
