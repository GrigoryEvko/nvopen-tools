// Function: sub_3860240
// Address: 0x3860240
//
__int64 __fastcall sub_3860240(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, int a5, __int64 a6)
{
  unsigned int v8; // edi
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  unsigned int v11; // ecx
  __int64 *v12; // rax
  unsigned int *v13; // r15
  unsigned int *v14; // rbx
  __int64 v15; // rdx
  unsigned int *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rax
  int v21; // eax
  int v22; // r10d

  v8 = *(_DWORD *)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 24);
  if ( v8 )
  {
    a5 = v8 - 1;
    v10 = (4LL * a4) | a3 & 0xFFFFFFFFFFFFFFFBLL;
    v11 = (v8 - 1) & (v10 ^ (v10 >> 9));
    v12 = (__int64 *)(v9 + 32LL * v11);
    a6 = *v12;
    if ( v10 == *v12 )
      goto LABEL_3;
    v21 = 1;
    while ( a6 != -4 )
    {
      v22 = v21 + 1;
      v11 = a5 & (v21 + v11);
      v12 = (__int64 *)(v9 + 32LL * v11);
      a6 = *v12;
      if ( v10 == *v12 )
        goto LABEL_3;
      v21 = v22;
    }
  }
  v12 = (__int64 *)(v9 + 32LL * v8);
LABEL_3:
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v13 = (unsigned int *)v12[2];
  v14 = (unsigned int *)v12[1];
  if ( v13 != v14 )
  {
    v15 = *v14;
    v16 = v14 + 1;
    v17 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v15);
    v18 = a1 + 16;
    v19 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v18 + 8 * v19) = v17;
      v19 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v19;
      if ( v16 == v13 )
        break;
      v17 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL * *v16);
      if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v19 )
      {
        sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, a5, a6);
        v19 = *(unsigned int *)(a1 + 8);
      }
      v18 = *(_QWORD *)a1;
      ++v16;
    }
  }
  return a1;
}
