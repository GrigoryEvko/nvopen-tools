// Function: sub_31F3F30
// Address: 0x31f3f30
//
void __fastcall sub_31F3F30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  int v10; // r13d
  __int64 v11; // rdx
  const void *v12; // rsi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // rdi
  __int64 v20; // rsi

  if ( a1 != a2 )
  {
    v8 = *(unsigned int *)(a2 + 8);
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_DWORD *)(a2 + 8);
    if ( v8 > v9 )
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v8, 0x10u, a5, a6);
        v8 = *(unsigned int *)(a2 + 8);
        v9 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v17 = *(__int64 **)a2;
        v9 *= 16LL;
        v18 = *(_QWORD *)a1;
        v19 = (__int64 *)(*(_QWORD *)a2 + v9);
        do
        {
          v20 = *v17;
          v17 += 2;
          v18 += 16;
          *(_QWORD *)(v18 - 16) = v20;
          *(_QWORD *)(v18 - 8) = *(v17 - 1);
        }
        while ( v17 != v19 );
        v8 = *(unsigned int *)(a2 + 8);
      }
      v11 = 16 * v8;
      v12 = (const void *)(*(_QWORD *)a2 + v9);
      if ( v12 != (const void *)(v11 + *(_QWORD *)a2) )
        memcpy((void *)(v9 + *(_QWORD *)a1), v12, v11 - v9);
      goto LABEL_7;
    }
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_7:
      *(_DWORD *)(a1 + 8) = v10;
      return;
    }
    v13 = *(__int64 **)a2;
    v14 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a2 + 16 * v8;
    do
    {
      v16 = *v13;
      v13 += 2;
      v14 += 16;
      *(_QWORD *)(v14 - 16) = v16;
      *(_QWORD *)(v14 - 8) = *(v13 - 1);
    }
    while ( v13 != (__int64 *)v15 );
    *(_DWORD *)(a1 + 8) = v10;
  }
}
