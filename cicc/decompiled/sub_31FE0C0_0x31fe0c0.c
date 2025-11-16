// Function: sub_31FE0C0
// Address: 0x31fe0c0
//
__int64 __fastcall sub_31FE0C0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 result; // rax
  __int64 v8; // r13
  char v10; // al
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r13
  __int64 v17; // r12
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  bool v21; // cc
  unsigned __int64 v22; // rdi

  v6 = *a1;
  result = 11LL * *((unsigned int *)a1 + 2);
  v8 = (__int64)&(*a1)[11 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (__int64 *)v8 )
  {
    do
    {
      if ( a2 )
      {
        v12 = *v6;
        *(_DWORD *)(a2 + 32) = 0;
        *(_QWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 24) = 0;
        *(_DWORD *)(a2 + 28) = 0;
        *(_QWORD *)a2 = v12;
        *(_QWORD *)(a2 + 8) = 1;
        v13 = v6[2];
        ++v6[1];
        v14 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(a2 + 16) = v13;
        LODWORD(v13) = *((_DWORD *)v6 + 6);
        v6[2] = v14;
        LODWORD(v14) = *(_DWORD *)(a2 + 24);
        *(_DWORD *)(a2 + 24) = v13;
        LODWORD(v13) = *((_DWORD *)v6 + 7);
        *((_DWORD *)v6 + 6) = v14;
        LODWORD(v14) = *(_DWORD *)(a2 + 28);
        *(_DWORD *)(a2 + 28) = v13;
        v15 = *((unsigned int *)v6 + 8);
        *((_DWORD *)v6 + 7) = v14;
        LODWORD(v14) = *(_DWORD *)(a2 + 32);
        *(_DWORD *)(a2 + 32) = v15;
        *((_DWORD *)v6 + 8) = v14;
        *(_QWORD *)(a2 + 40) = a2 + 56;
        *(_DWORD *)(a2 + 48) = 0;
        *(_DWORD *)(a2 + 52) = 0;
        if ( *((_DWORD *)v6 + 12) )
          sub_31FDD40(a2 + 40, (__int64)(v6 + 5), v15, a4, a5, a6);
        v10 = *((_BYTE *)v6 + 56);
        *(_BYTE *)(a2 + 80) = 0;
        *(_BYTE *)(a2 + 56) = v10;
        if ( *((_BYTE *)v6 + 80) )
        {
          *(_DWORD *)(a2 + 72) = *((_DWORD *)v6 + 18);
          *(_QWORD *)(a2 + 64) = v6[8];
          v11 = *((_BYTE *)v6 + 76);
          *((_DWORD *)v6 + 18) = 0;
          *(_BYTE *)(a2 + 76) = v11;
          *(_BYTE *)(a2 + 80) = 1;
        }
      }
      v6 += 11;
      a2 += 88;
    }
    while ( (__int64 *)v8 != v6 );
    v16 = *a1;
    result = 11LL * *((unsigned int *)a1 + 2);
    v17 = (__int64)&(*a1)[11 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (__int64 *)v17 )
    {
      do
      {
        v17 -= 88;
        if ( *(_BYTE *)(v17 + 80) )
        {
          v21 = *(_DWORD *)(v17 + 72) <= 0x40u;
          *(_BYTE *)(v17 + 80) = 0;
          if ( !v21 )
          {
            v22 = *(_QWORD *)(v17 + 64);
            if ( v22 )
              j_j___libc_free_0_0(v22);
          }
        }
        v18 = *(_QWORD *)(v17 + 40);
        v19 = v18 + 40LL * *(unsigned int *)(v17 + 48);
        if ( v18 != v19 )
        {
          do
          {
            v19 -= 40LL;
            v20 = *(_QWORD *)(v19 + 8);
            if ( v20 != v19 + 24 )
              _libc_free(v20);
          }
          while ( v18 != v19 );
          v18 = *(_QWORD *)(v17 + 40);
        }
        if ( v18 != v17 + 56 )
          _libc_free(v18);
        result = sub_C7D6A0(*(_QWORD *)(v17 + 16), 12LL * *(unsigned int *)(v17 + 32), 4);
      }
      while ( (__int64 *)v17 != v16 );
    }
  }
  return result;
}
