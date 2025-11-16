// Function: sub_1A7EC10
// Address: 0x1a7ec10
//
void __fastcall sub_1A7EC10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  int v10; // r13d
  __int64 v11; // rdx
  const void *v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rdi
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
        sub_16CD150(a1, (const void *)(a1 + 16), v8, 16, a5, a6);
        v8 = *(unsigned int *)(a2 + 8);
        v9 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v17 = *(_QWORD **)a2;
        v9 *= 16LL;
        v18 = *(_QWORD *)a1;
        v19 = (_QWORD *)(*(_QWORD *)a2 + v9);
        do
        {
          v20 = *v17;
          v17 += 2;
          v18 += 16;
          *(_QWORD *)(v18 - 16) = v20;
          *(_DWORD *)(v18 - 8) = *((_DWORD *)v17 - 2);
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
    v13 = *(_QWORD **)a2;
    v14 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a2 + 16 * v8;
    do
    {
      v16 = *v13;
      v13 += 2;
      v14 += 16;
      *(_QWORD *)(v14 - 16) = v16;
      *(_DWORD *)(v14 - 8) = *((_DWORD *)v13 - 2);
    }
    while ( v13 != (_QWORD *)v15 );
    *(_DWORD *)(a1 + 8) = v10;
  }
}
