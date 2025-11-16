// Function: sub_3763120
// Address: 0x3763120
//
__int64 __fastcall sub_3763120(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  _QWORD *v9; // rax
  int v10; // r8d
  const void *v11; // r9
  __int64 result; // rax
  int v13; // eax
  __int64 v14; // rsi
  int v15; // edx
  unsigned int v16; // eax
  __int64 *v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rdi
  int v21; // ecx
  int v22; // r8d
  unsigned __int64 v23; // [rsp+0h] [rbp-30h] BYREF
  __int64 v24[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a1 + 24);
  v24[0] = a2;
  v23 = a3;
  sub_375DA40(v5, a2, a3);
  v6 = *(_QWORD *)(a1 + 32);
  if ( !*(_DWORD *)(v6 + 16) )
  {
    v7 = *(_QWORD **)(v6 + 32);
    v8 = (__int64)&v7[*(unsigned int *)(v6 + 40)];
    v9 = sub_3759330(v7, v8, v24);
    if ( (_QWORD *)v8 == v9 )
      goto LABEL_6;
    v11 = v9 + 1;
    if ( (_QWORD *)v8 == v9 + 1 )
      goto LABEL_5;
LABEL_4:
    memmove(v9, v11, v8 - (_QWORD)v11);
    v10 = *(_DWORD *)(v6 + 40);
LABEL_5:
    *(_DWORD *)(v6 + 40) = v10 - 1;
    goto LABEL_6;
  }
  v13 = *(_DWORD *)(v6 + 24);
  v14 = *(_QWORD *)(v6 + 8);
  if ( v13 )
  {
    v15 = v13 - 1;
    v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 *)(v14 + 8LL * v16);
    v18 = *v17;
    if ( a2 == *v17 )
    {
LABEL_10:
      *v17 = -8192;
      v19 = *(unsigned int *)(v6 + 40);
      --*(_DWORD *)(v6 + 16);
      v20 = *(_QWORD **)(v6 + 32);
      ++*(_DWORD *)(v6 + 20);
      v8 = (__int64)&v20[v19];
      v9 = sub_3759330(v20, v8, v24);
      v11 = v9 + 1;
      if ( v9 + 1 == (_QWORD *)v8 )
        goto LABEL_5;
      goto LABEL_4;
    }
    v21 = 1;
    while ( v18 != -4096 )
    {
      v22 = v21 + 1;
      v16 = v15 & (v21 + v16);
      v17 = (__int64 *)(v14 + 8LL * v16);
      v18 = *v17;
      if ( a2 == *v17 )
        goto LABEL_10;
      v21 = v22;
    }
  }
LABEL_6:
  result = v23;
  if ( *(_DWORD *)(v23 + 36) == -1 )
    return sub_3762BC0(*(_QWORD *)(a1 + 32), (__int64 *)&v23);
  return result;
}
