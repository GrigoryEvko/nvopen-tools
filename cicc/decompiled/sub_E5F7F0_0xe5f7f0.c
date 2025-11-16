// Function: sub_E5F7F0
// Address: 0xe5f7f0
//
__int64 __fastcall sub_E5F7F0(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  int v7; // eax
  unsigned int v8; // r8d
  __int64 *v9; // r10
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rcx
  int v13; // eax
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 *v17; // r10
  __int64 v18; // rcx
  __int64 **v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rdx
  __int64 v23; // r12
  int v24; // eax
  const void *v25; // r14
  __int64 v26; // rdi
  size_t v27; // r12
  __int64 v28; // [rsp+0h] [rbp-70h]
  __int64 *v29; // [rsp+8h] [rbp-68h]
  unsigned int v30; // [rsp+14h] [rbp-5Ch]
  __int64 v31; // [rsp+18h] [rbp-58h]

  v31 = *(_QWORD *)(a2 + 48);
  v7 = sub_C92610();
  v8 = sub_C92740(a2 + 8, a3, a4, v7);
  v9 = (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * v8);
  v10 = (__int64 *)*v9;
  if ( !*v9 )
  {
LABEL_6:
    v29 = v9;
    v30 = v8;
    v15 = sub_C7D670(a4 + 17, 8);
    v16 = v30;
    v17 = v29;
    v18 = v15;
    if ( a4 )
    {
      v28 = v15;
      memcpy((void *)(v15 + 16), a3, a4);
      v16 = v30;
      v17 = v29;
      v18 = v28;
    }
    *(_BYTE *)(v18 + a4 + 16) = 0;
    *(_QWORD *)v18 = a4;
    *(_DWORD *)(v18 + 8) = v31;
    *v17 = v18;
    ++*(_DWORD *)(a2 + 20);
    v19 = (__int64 **)(*(_QWORD *)(a2 + 8) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a2 + 8), v16));
    v22 = *v19;
    if ( *v19 )
      goto LABEL_10;
    do
    {
      do
      {
        v22 = v19[1];
        ++v19;
      }
      while ( !v22 );
LABEL_10:
      ;
    }
    while ( v22 == (__int64 *)-8LL );
    v23 = *v22;
    v24 = *((_DWORD *)v22 + 2);
    v25 = v22 + 2;
    v26 = *(_QWORD *)(a2 + 48);
    *(_QWORD *)a1 = v22 + 2;
    *(_QWORD *)(a1 + 8) = v23;
    v27 = v23 + 1;
    *(_DWORD *)(a1 + 16) = v24;
    if ( v27 + v26 > *(_QWORD *)(a2 + 56) )
    {
      sub_C8D290(a2 + 40, (const void *)(a2 + 64), v27 + v26, 1u, v20, v21);
      v26 = *(_QWORD *)(a2 + 48);
      if ( !v27 )
        goto LABEL_14;
    }
    else if ( !v27 )
    {
LABEL_14:
      *(_QWORD *)(a2 + 48) = v26 + v27;
      return a1;
    }
    memcpy((void *)(*(_QWORD *)(a2 + 40) + v26), v25, v27);
    v26 = *(_QWORD *)(a2 + 48);
    goto LABEL_14;
  }
  if ( v10 == (__int64 *)-8LL )
  {
    --*(_DWORD *)(a2 + 24);
    goto LABEL_6;
  }
  v11 = *v10;
  v12 = v10 + 2;
  v13 = *((_DWORD *)v10 + 2);
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = v11;
  *(_DWORD *)(a1 + 16) = v13;
  return a1;
}
