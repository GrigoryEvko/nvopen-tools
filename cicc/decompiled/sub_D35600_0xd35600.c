// Function: sub_D35600
// Address: 0xd35600
//
__int64 __fastcall sub_D35600(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  unsigned int v9; // edi
  __int64 v10; // rsi
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

  v6 = (4LL * a4) | a3 & 0xFFFFFFFFFFFFFFFBLL;
  v9 = *(_DWORD *)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 32);
  if ( v9 )
  {
    a5 = v9 - 1;
    v11 = a5 & (v6 ^ (v6 >> 9));
    v12 = (__int64 *)(v10 + 32LL * v11);
    a6 = *v12;
    if ( v6 == *v12 )
      goto LABEL_3;
    v21 = 1;
    while ( a6 != -4 )
    {
      v22 = v21 + 1;
      v11 = a5 & (v21 + v11);
      v12 = (__int64 *)(v10 + 32LL * v11);
      a6 = *v12;
      if ( v6 == *v12 )
        goto LABEL_3;
      v21 = v22;
    }
  }
  v12 = (__int64 *)(v10 + 32LL * v9);
LABEL_3:
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v13 = (unsigned int *)v12[2];
  v14 = (unsigned int *)v12[1];
  if ( v13 != v14 )
  {
    v15 = *v14;
    v16 = v14 + 1;
    v17 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8 * v15);
    v18 = a1 + 16;
    v19 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v18 + 8 * v19) = v17;
      v19 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v19;
      if ( v13 == v16 )
        break;
      v17 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8LL * *v16);
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v19 + 1, 8u, a5, a6);
        v19 = *(unsigned int *)(a1 + 8);
      }
      v18 = *(_QWORD *)a1;
      ++v16;
    }
  }
  return a1;
}
