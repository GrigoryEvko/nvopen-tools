// Function: sub_1D236A0
// Address: 0x1d236a0
//
__int64 __fastcall sub_1D236A0(__int64 a1)
{
  unsigned int v2; // r12d
  __int64 *v3; // r14
  __int64 *v4; // rbx
  unsigned __int64 *v5; // rdi
  unsigned __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 *v10; // r13
  int v11; // edx
  __int64 *i; // r13
  __int64 j; // rcx
  __int64 v14; // rdx
  unsigned __int64 *v15; // r9
  unsigned __int64 v16; // r8
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 *v22; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v3 = *(__int64 **)(a1 + 200);
  v22 = (__int64 *)(a1 + 192);
  v4 = v3;
  if ( (__int64 *)(a1 + 192) != v3 )
  {
    do
    {
      while ( 1 )
      {
        v10 = v3;
        v3 = (__int64 *)v3[1];
        nullsub_686();
        v11 = *((_DWORD *)v10 + 12);
        if ( !v11 )
          break;
        *((_DWORD *)v10 + 5) = v11;
        if ( v22 == v3 )
          goto LABEL_8;
      }
      *((_DWORD *)v10 + 5) = v2;
      if ( v10 != v4 )
      {
        v5 = (unsigned __int64 *)v10[1];
        v6 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
        *v5 = v6 | *v5 & 7;
        *(_QWORD *)(v6 + 8) = v5;
        v7 = *v10;
        v10[1] = 0;
        v8 = v7 & 7;
        *v10 = v8;
        v9 = *v4;
        v10[1] = (__int64)v4;
        v9 &= 0xFFFFFFFFFFFFFFF8LL;
        *v10 = v9 | v8;
        *(_QWORD *)(v9 + 8) = v10;
        *v4 = (unsigned __int64)v10 | *v4 & 7;
      }
      v4 = (__int64 *)v10[1];
      ++v2;
    }
    while ( v22 != v3 );
LABEL_8:
    for ( i = *(__int64 **)(a1 + 200); v3 != i; i = (__int64 *)i[1] )
    {
      if ( !i )
        JUMPOUT(0x42097A);
      nullsub_686();
      for ( j = i[5]; j; ++v2 )
      {
        while ( 1 )
        {
          v20 = *(_QWORD *)(j + 16);
          if ( *(_DWORD *)(v20 + 28) == 1 )
            break;
          --*(_DWORD *)(v20 + 28);
          j = *(_QWORD *)(j + 32);
          if ( !j )
            goto LABEL_17;
        }
        v14 = v20 + 8;
        *(_DWORD *)(v20 + 28) = v2;
        if ( (__int64 *)(v20 + 8) != v4 )
        {
          v15 = *(unsigned __int64 **)(v20 + 16);
          v16 = *(_QWORD *)(v20 + 8) & 0xFFFFFFFFFFFFFFF8LL;
          *v15 = v16 | *v15 & 7;
          *(_QWORD *)(v16 + 8) = v15;
          v17 = *(_QWORD *)(v20 + 8);
          *(_QWORD *)(v20 + 16) = 0;
          v18 = v17 & 7;
          *(_QWORD *)(v20 + 8) = v18;
          v19 = *v4;
          *(_QWORD *)(v20 + 16) = v4;
          v19 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v20 + 8) = v19 | v18;
          *(_QWORD *)(v19 + 8) = v14;
          *v4 = *v4 & 7 | v14;
        }
        j = *(_QWORD *)(j + 32);
        v4 = *(__int64 **)(v20 + 16);
      }
LABEL_17:
      ;
    }
  }
  return v2;
}
