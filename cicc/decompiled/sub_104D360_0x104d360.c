// Function: sub_104D360
// Address: 0x104d360
//
bool __fastcall sub_104D360(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ecx
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 *v16; // r14
  int v18; // eax
  int v19; // r10d
  __int64 v20; // rax

  v4 = *(_DWORD *)(a1 + 600);
  v5 = *(_QWORD *)(a1 + 584);
  v6 = *(_QWORD *)(a3 + 40);
  if ( v4 )
  {
    v7 = (v4 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( v6 == *v8 )
      goto LABEL_3;
    v18 = 1;
    while ( v9 != -4096 )
    {
      v19 = v18 + 1;
      v20 = (v4 - 1) & (v7 + v18);
      v7 = v20;
      v8 = (__int64 *)(v5 + 16 * v20);
      v9 = *v8;
      if ( v6 == *v8 )
        goto LABEL_3;
      v18 = v19;
    }
  }
  v8 = (__int64 *)(v5 + 16LL * v4);
LABEL_3:
  v10 = *(_QWORD *)(a1 + 48);
  v11 = 8LL * *((unsigned int *)v8 + 2) + 8;
  v12 = (__int64 *)(v10 + v11);
  v13 = 8LL * *((unsigned int *)v8 + 3) - v11;
  v14 = v13 >> 3;
  if ( v13 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v15 = v14 >> 1;
        v16 = &v12[v14 >> 1];
        if ( sub_B445A0(a3, *v16) )
          break;
        v12 = v16 + 1;
        v14 = v14 - v15 - 1;
        if ( v14 <= 0 )
          goto LABEL_8;
      }
      v14 >>= 1;
    }
    while ( v15 > 0 );
LABEL_8:
    v10 = *(_QWORD *)(a1 + 48);
  }
  return (*(_QWORD *)(*(_QWORD *)sub_104D250(a1, a2) + 8LL * ((unsigned int)(((__int64)v12 - v10 - 8) >> 3) >> 6))
        & (1LL << (((__int64)v12 - v10 - 8) >> 3))) != 0;
}
