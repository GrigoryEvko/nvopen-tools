// Function: sub_2B1F880
// Address: 0x2b1f880
//
char __fastcall sub_2B1F880(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v8; // r15
  unsigned __int64 v9; // rdx
  __int64 v10; // rdi
  const void *v11; // r8
  unsigned __int64 v12; // r13
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  int v15; // eax
  unsigned __int64 v16; // r10
  __int64 v17; // rdi
  int v18; // r8d
  unsigned int v19; // r13d
  __int64 v20; // r9
  __int64 v21; // rax
  int v22; // edx
  int *v23; // rsi
  int v24; // ecx
  int v26; // r12d
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-40h]
  int v29; // [rsp+8h] [rbp-38h]
  const void *v30; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)a2 != 92 || *(_BYTE *)a3 != 92 )
    return sub_B46220(a2, a3);
  v8 = sub_B46220(a2, a3);
  if ( v8 || *(_QWORD *)(a2 - 64) != *(_QWORD *)(a3 - 64) || *(_QWORD *)(a2 - 32) != *(_QWORD *)(a3 - 32) )
    return v8;
  v14 = *(unsigned int *)(a3 + 80);
  v9 = *(unsigned int *)(a4 + 12);
  v10 = 0;
  v11 = *(const void **)(a3 + 72);
  *(_DWORD *)(a4 + 8) = 0;
  v12 = v14;
  v13 = 4 * v14;
  LODWORD(v14) = 0;
  if ( v12 > v9 )
  {
    v28 = v13;
    v30 = v11;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v12, 4u, (__int64)v11, v13);
    v14 = *(unsigned int *)(a4 + 8);
    v13 = v28;
    v11 = v30;
    v10 = 4 * v14;
  }
  if ( v13 )
  {
    memcpy((void *)(*(_QWORD *)a4 + v10), v11, v13);
    LODWORD(v14) = *(_DWORD *)(a4 + 8);
  }
  v15 = v12 + v14;
  *(_DWORD *)(a4 + 8) = v15;
  v16 = *(unsigned int *)(a2 + 80);
  v17 = *(_QWORD *)(a2 + 72);
  v18 = *(_DWORD *)(a2 + 80);
  if ( v15 <= 0 )
  {
    v19 = 0;
    goto LABEL_20;
  }
  v19 = 0;
  v20 = 4LL * (unsigned int)(v15 - 1) + 4;
  v21 = 0;
  do
  {
    while ( 1 )
    {
      v22 = *(_DWORD *)(v17 + v21);
      v23 = (int *)(v21 + *(_QWORD *)a4);
      v24 = *v23;
      if ( v22 == -1 )
        break;
      v19 = 0;
      if ( v24 == -1 )
        goto LABEL_18;
      if ( v24 != v22 )
        return v8;
      v19 = 0;
LABEL_15:
      v21 += 4;
      if ( v20 == v21 )
        goto LABEL_19;
    }
    ++v19;
    if ( v24 != -1 )
      goto LABEL_15;
LABEL_18:
    v21 += 4;
    *v23 = v22;
  }
  while ( v20 != v21 );
LABEL_19:
  v16 -= v19;
LABEL_20:
  v29 = v18;
  if ( v16 > 1 )
  {
    v26 = sub_2B1F810(*a1, *(_QWORD *)(a2 + 8), 0xFFFFFFFF);
    v27 = sub_2B08680(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL), v29 - v19);
    return v26 == (unsigned int)sub_2B1F810(*a1, v27, 0xFFFFFFFF);
  }
  return v8;
}
