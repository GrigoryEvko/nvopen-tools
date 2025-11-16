// Function: sub_CCDFB0
// Address: 0xccdfb0
//
bool __fastcall sub_CCDFB0(unsigned int *a1, __int64 a2)
{
  const void *v3; // rdi
  const void *v5; // rsi
  bool v6; // r13
  size_t v7; // rdx
  const void *v8; // rdi
  const void *v9; // rsi
  bool v10; // r14
  __int64 v11; // rdx
  const void *v12; // rdi
  const void *v13; // rsi
  bool result; // al
  __int64 v15; // rdx
  bool v16; // zf

  v3 = (const void *)*((_QWORD *)a1 + 1);
  v5 = *(const void **)(a2 + 8);
  if ( !v3 )
  {
    v8 = (const void *)*((_QWORD *)a1 + 4);
    v16 = v5 == 0;
    v9 = *(const void **)(a2 + 32);
    v6 = v16;
    if ( v8 )
      goto LABEL_6;
LABEL_14:
    v12 = (const void *)*((_QWORD *)a1 + 5);
    v16 = v9 == 0;
    v13 = *(const void **)(a2 + 40);
    v10 = v16;
    result = 0;
    if ( v12 )
      goto LABEL_10;
LABEL_15:
    if ( v13 )
      return result;
    result = v10 && v6;
    if ( !v10 || !v6 )
      return result;
    result = 0;
    if ( *a1 != *(_DWORD *)a2 )
      return result;
    goto LABEL_20;
  }
  v6 = 0;
  if ( v5 )
  {
    v7 = *a1;
    if ( (_DWORD)v7 == *(_DWORD *)a2 )
      v6 = memcmp(v3, v5, v7) != 0;
  }
  v8 = (const void *)*((_QWORD *)a1 + 4);
  v9 = *(const void **)(a2 + 32);
  if ( !v8 )
    goto LABEL_14;
LABEL_6:
  v10 = 0;
  if ( v9 )
  {
    v11 = *a1;
    if ( (_DWORD)v11 == *(_DWORD *)a2 )
      v10 = memcmp(v8, v9, 4 * v11) != 0;
  }
  v12 = (const void *)*((_QWORD *)a1 + 5);
  v13 = *(const void **)(a2 + 40);
  result = 0;
  if ( !v12 )
    goto LABEL_15;
LABEL_10:
  if ( !v13 )
    return result;
  v15 = *a1;
  if ( (_DWORD)v15 != *(_DWORD *)a2 )
    return result;
  result = memcmp(v12, v13, 4 * v15) != 0 && v10 && v6;
  if ( !result )
    return result;
LABEL_20:
  result = 0;
  if ( ((*(_BYTE *)(a2 + 16) ^ *((_BYTE *)a1 + 16)) & 7) == 0 && a1[5] == *(_DWORD *)(a2 + 20) )
    return a1[6] == *(_DWORD *)(a2 + 24);
  return result;
}
