// Function: sub_2E79610
// Address: 0x2e79610
//
_QWORD *__fastcall sub_2E79610(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r9
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // rax
  unsigned int v11; // esi
  __int64 *v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // edx
  int v16; // r12d

  v5 = *(_QWORD *)(a2 + 688);
  v6 = *(_QWORD *)(a2 + 696);
  v7 = *(unsigned int *)(a2 + 712);
  v8 = a2 + 688;
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 8) + 904LL) & 1) == 0 )
  {
    *a1 = v8;
    v9 = v6 + 32 * v7;
    a1[1] = v5;
    a1[2] = v9;
    a1[3] = v9;
    return a1;
  }
  if ( !(_DWORD)v7 )
    goto LABEL_7;
  v11 = (v7 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v6 + 32LL * v11);
  v13 = *v12;
  if ( a3 != *v12 )
  {
    v15 = 1;
    while ( v13 != -4096 )
    {
      v16 = v15 + 1;
      v11 = (v7 - 1) & (v15 + v11);
      v12 = (__int64 *)(v6 + 32LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        goto LABEL_5;
      v15 = v16;
    }
LABEL_7:
    *a1 = v8;
    v14 = v6 + 32 * v7;
    a1[1] = v5;
    a1[2] = v14;
    a1[3] = v14;
    return a1;
  }
LABEL_5:
  *a1 = v8;
  a1[1] = v5;
  a1[2] = v12;
  a1[3] = v6 + 32 * v7;
  return a1;
}
