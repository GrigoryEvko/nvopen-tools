// Function: sub_37FA100
// Address: 0x37fa100
//
_QWORD *__fastcall sub_37FA100(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int16 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  size_t v14; // rdx
  size_t v15; // r12
  const void *v16; // r15
  unsigned __int64 v17; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax

  v6 = a2 + 24;
  v9 = *(_WORD *)(a4 + 6);
  if ( (v9 & 1) != 0 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(v19 + 6) > *(_QWORD *)(a2 + 40) )
    {
      sub_C8D290(v6, (const void *)(a2 + 48), v19 + 6, 1u, a5, a6);
      v19 = *(_QWORD *)(a2 + 32);
    }
    v20 = *(_QWORD *)(a2 + 24) + v19;
    *(_DWORD *)v20 = 1936617315;
    *(_WORD *)(v20 + 4) = 8308;
    *(_QWORD *)(a2 + 32) += 6LL;
    if ( (v9 & 2) == 0 )
    {
LABEL_3:
      if ( (v9 & 4) == 0 )
        goto LABEL_4;
      goto LABEL_15;
    }
  }
  else if ( (v9 & 2) == 0 )
  {
    goto LABEL_3;
  }
  v21 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(v21 + 9) > *(_QWORD *)(a2 + 40) )
  {
    sub_C8D290(v6, (const void *)(a2 + 48), v21 + 9, 1u, a5, a6);
    v21 = *(_QWORD *)(a2 + 32);
  }
  v22 = *(_QWORD *)(a2 + 24) + v21;
  *(_QWORD *)v22 = 0x656C6974616C6F76LL;
  *(_BYTE *)(v22 + 8) = 32;
  *(_QWORD *)(a2 + 32) += 9LL;
  if ( (v9 & 4) != 0 )
  {
LABEL_15:
    v23 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(v23 + 12) > *(_QWORD *)(a2 + 40) )
    {
      sub_C8D290(v6, (const void *)(a2 + 48), v23 + 12, 1u, a5, a6);
      v23 = *(_QWORD *)(a2 + 32);
    }
    qmemcpy((void *)(*(_QWORD *)(a2 + 24) + v23), "__unaligned ", 12);
    *(_QWORD *)(a2 + 32) += 12LL;
  }
LABEL_4:
  v10 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 40LL))(
          *(_QWORD *)(a2 + 8),
          *(unsigned int *)(a4 + 2));
  v13 = *(_QWORD *)(a2 + 32);
  v15 = v14;
  v16 = (const void *)v10;
  v17 = v13 + v14;
  if ( v17 > *(_QWORD *)(a2 + 40) )
  {
    sub_C8D290(v6, (const void *)(a2 + 48), v17, 1u, v11, v12);
    v13 = *(_QWORD *)(a2 + 32);
  }
  if ( v15 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 24) + v13), v16, v15);
    v13 = *(_QWORD *)(a2 + 32);
  }
  *(_QWORD *)(a2 + 32) = v13 + v15;
  *a1 = 1;
  return a1;
}
