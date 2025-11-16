// Function: sub_2DF64A0
// Address: 0x2df64a0
//
__int64 __fastcall sub_2DF64A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v7; // edx
  int v8; // ecx
  unsigned int v9; // eax
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 result; // rax
  unsigned int v14; // r12d
  __int64 v15; // rdx
  unsigned __int64 *v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r14
  __int64 v20; // r15

  v6 = *a1;
  v7 = *((_DWORD *)a1 + 5);
  v8 = *(_DWORD *)(*a1 + 160);
  v9 = *(_DWORD *)(*a1 + 164);
  *((_DWORD *)a1 + 4) = 0;
  if ( !v8 )
  {
    v10 = v9;
    v11 = 0;
    if ( v7 )
      goto LABEL_3;
LABEL_6:
    sub_C8D5F0((__int64)(a1 + 1), a1 + 3, 1u, 0x10u, a5, a6);
    v11 = 16LL * *((unsigned int *)a1 + 4);
    goto LABEL_3;
  }
  v6 += 8;
  v10 = v9;
  v11 = 0;
  if ( !v7 )
    goto LABEL_6;
LABEL_3:
  v12 = a1[1];
  *(_QWORD *)(v12 + v11) = v6;
  *(_QWORD *)(v12 + v11 + 8) = v10;
  result = *a1;
  ++*((_DWORD *)a1 + 4);
  v14 = *(_DWORD *)(result + 160);
  if ( v14 )
  {
    v15 = *((unsigned int *)a1 + 4);
    for ( result = (unsigned int)(v15 - 1); v14 > (unsigned int)result; *((_DWORD *)a1 + 4) = v15 )
    {
      v17 = a1[1];
      v18 = v15 + 1;
      v19 = *(_QWORD *)(*(_QWORD *)(v17 + 16 * result) + 8LL * *(unsigned int *)(v17 + 16 * result + 12))
          & 0xFFFFFFFFFFFFFFC0LL;
      v20 = (*(_QWORD *)(*(_QWORD *)(v17 + 16 * result) + 8LL * *(unsigned int *)(v17 + 16 * result + 12)) & 0x3FLL) + 1;
      if ( v18 > *((unsigned int *)a1 + 5) )
      {
        sub_C8D5F0((__int64)(a1 + 1), a1 + 3, v18, 0x10u, a5, a6);
        v17 = a1[1];
      }
      v16 = (unsigned __int64 *)(v17 + 16LL * *((unsigned int *)a1 + 4));
      *v16 = v19;
      v16[1] = v20;
      result = *((unsigned int *)a1 + 4);
      v15 = (unsigned int)(result + 1);
    }
  }
  return result;
}
