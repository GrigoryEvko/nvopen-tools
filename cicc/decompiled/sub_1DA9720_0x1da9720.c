// Function: sub_1DA9720
// Address: 0x1da9720
//
__int64 __fastcall sub_1DA9720(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
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
  unsigned int v15; // ecx
  unsigned __int64 *v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 v18; // r14
  __int64 v19; // r15

  v6 = *a1;
  v7 = *((_DWORD *)a1 + 5);
  v8 = *(_DWORD *)(*a1 + 80);
  v9 = *(_DWORD *)(*a1 + 84);
  *((_DWORD *)a1 + 4) = 0;
  if ( !v8 )
  {
    v10 = v9;
    v11 = 0;
    if ( v7 )
      goto LABEL_3;
LABEL_6:
    sub_16CD150((__int64)(a1 + 1), a1 + 3, 0, 16, a5, a6);
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
  v14 = *(_DWORD *)(result + 80);
  if ( v14 )
  {
    v15 = *((_DWORD *)a1 + 4);
    for ( result = v15 - 1; v14 > (unsigned int)result; *((_DWORD *)a1 + 4) = result + 1 )
    {
      v17 = a1[1];
      v18 = *(_QWORD *)(*(_QWORD *)(v17 + 16 * result) + 8LL * *(unsigned int *)(v17 + 16 * result + 12))
          & 0xFFFFFFFFFFFFFFC0LL;
      v19 = (*(_QWORD *)(*(_QWORD *)(v17 + 16 * result) + 8LL * *(unsigned int *)(v17 + 16 * result + 12)) & 0x3FLL) + 1;
      if ( v15 >= *((_DWORD *)a1 + 5) )
      {
        sub_16CD150((__int64)(a1 + 1), a1 + 3, 0, 16, a5, a6);
        v17 = a1[1];
      }
      v16 = (unsigned __int64 *)(16LL * *((unsigned int *)a1 + 4) + v17);
      *v16 = v18;
      v16[1] = v19;
      result = *((unsigned int *)a1 + 4);
      v15 = result + 1;
    }
  }
  return result;
}
