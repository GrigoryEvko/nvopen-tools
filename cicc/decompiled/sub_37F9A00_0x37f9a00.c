// Function: sub_37F9A00
// Address: 0x37f9a00
//
_QWORD *__fastcall sub_37F9A00(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned int *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rax
  size_t v15; // rdx
  size_t v16; // rbx
  __int64 v17; // rdx
  const void *v18; // r11
  const void *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  unsigned int *v22; // [rsp+18h] [rbp-38h]

  v6 = 0;
  v7 = a2 + 24;
  v8 = *(unsigned int **)(a4 + 8);
  v9 = *(_QWORD *)(a4 + 16);
  *(_QWORD *)(a2 + 32) = 0;
  v10 = (v9 - (__int64)v8) >> 2;
  if ( !*(_QWORD *)(a2 + 40) )
  {
    sub_C8D290(v7, (const void *)(a2 + 48), 1, 1u, a5, a6);
    v6 = *(_QWORD *)(a2 + 32);
  }
  *(_BYTE *)(*(_QWORD *)(a2 + 24) + v6) = 34;
  v11 = *(_QWORD *)(a2 + 32) + 1LL;
  *(_QWORD *)(a2 + 32) = v11;
  if ( (_DWORD)v10 )
  {
    v12 = (unsigned int)(v10 - 1);
    v21 = (__int64)&v8[v12 + 1];
    v22 = &v8[v12];
    do
    {
      v14 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 40LL))(*(_QWORD *)(a2 + 8), *v8);
      v16 = v15;
      v17 = *(_QWORD *)(a2 + 32);
      v18 = (const void *)v14;
      if ( v16 + v17 > *(_QWORD *)(a2 + 40) )
      {
        v20 = (const void *)v14;
        sub_C8D290(v7, (const void *)(a2 + 48), v16 + v17, 1u, v16 + v17, a6);
        v17 = *(_QWORD *)(a2 + 32);
        v18 = v20;
      }
      if ( v16 )
      {
        memcpy((void *)(v17 + *(_QWORD *)(a2 + 24)), v18, v16);
        v17 = *(_QWORD *)(a2 + 32);
      }
      v11 = v16 + v17;
      *(_QWORD *)(a2 + 32) = v11;
      if ( v22 == v8 )
        break;
      if ( (unsigned __int64)(v11 + 3) > *(_QWORD *)(a2 + 40) )
      {
        sub_C8D290(v7, (const void *)(a2 + 48), v11 + 3, 1u, v11 + 3, a6);
        v11 = *(_QWORD *)(a2 + 32);
      }
      v13 = *(_QWORD *)(a2 + 24) + v11;
      ++v8;
      *(_WORD *)v13 = 8226;
      *(_BYTE *)(v13 + 2) = 34;
      v11 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v11;
    }
    while ( (unsigned int *)v21 != v8 );
  }
  if ( (unsigned __int64)(v11 + 1) > *(_QWORD *)(a2 + 40) )
  {
    sub_C8D290(v7, (const void *)(a2 + 48), v11 + 1, 1u, v11 + 1, a6);
    v11 = *(_QWORD *)(a2 + 32);
  }
  *(_BYTE *)(*(_QWORD *)(a2 + 24) + v11) = 34;
  ++*(_QWORD *)(a2 + 32);
  *a1 = 1;
  return a1;
}
