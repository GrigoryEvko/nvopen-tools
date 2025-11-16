// Function: sub_2916C30
// Address: 0x2916c30
//
unsigned __int64 __fastcall sub_2916C30(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  _QWORD *v9; // rdi
  __int64 *v11; // rsi
  unsigned __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  char v17; // dl
  __int64 v18; // r12
  unsigned __int64 v19; // rdx
  __int64 v20; // r12
  __int64 *v21; // r13
  char v22; // di
  __int64 v23; // rsi
  __int64 *v24; // r12

  v8 = *(unsigned int *)(a1 + 20);
  if ( (_DWORD)v8 != *(_DWORD *)(a1 + 24) )
  {
    v16 = *a2;
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_10;
    result = *(_QWORD *)(a1 + 8);
    a3 = (__int64 *)(result + 8LL * (unsigned int)v8);
    if ( (__int64 *)result != a3 )
    {
      while ( v16 != *(_QWORD *)result )
      {
        result += 8LL;
        if ( a3 == (__int64 *)result )
          goto LABEL_26;
      }
      return result;
    }
LABEL_26:
    if ( (unsigned int)v8 < *(_DWORD *)(a1 + 16) )
    {
      *(_DWORD *)(a1 + 20) = v8 + 1;
      *a3 = v16;
      ++*(_QWORD *)a1;
    }
    else
    {
LABEL_10:
      result = (unsigned __int64)sub_C8CC70(a1, v16, (__int64)a3, v8, a5, a6);
      if ( !v17 )
        return result;
    }
    result = *(unsigned int *)(a1 + 168);
    v18 = *a2;
    if ( result + 1 > *(unsigned int *)(a1 + 172) )
    {
      sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), result + 1, 8u, a5, a6);
      result = *(unsigned int *)(a1 + 168);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = v18;
    ++*(_DWORD *)(a1 + 168);
    return result;
  }
  v9 = *(_QWORD **)(a1 + 160);
  v11 = &v9[*(unsigned int *)(a1 + 168)];
  result = (unsigned __int64)sub_2912630(v9, (__int64)v11, a2);
  if ( v11 != (__int64 *)result )
    return result;
  v19 = v14 + 1;
  v20 = *a2;
  if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
  {
    sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v19, 8u, v14, v15);
    v19 = *(unsigned int *)(a1 + 168);
    v11 = (__int64 *)(*(_QWORD *)(a1 + 160) + 8 * v19);
  }
  *v11 = v20;
  result = (unsigned int)(*(_DWORD *)(a1 + 168) + 1);
  *(_DWORD *)(a1 + 168) = result;
  if ( (unsigned int)result <= 0x10 )
    return result;
  v21 = *(__int64 **)(a1 + 160);
  v22 = *(_BYTE *)(a1 + 28);
  v23 = *v21;
  v24 = &v21[result];
  if ( !v22 )
    goto LABEL_24;
LABEL_18:
  result = *(_QWORD *)(a1 + 8);
  v13 = *(unsigned int *)(a1 + 20);
  v19 = result + 8 * v13;
  if ( result != v19 )
  {
    while ( v23 != *(_QWORD *)result )
    {
      result += 8LL;
      if ( v19 == result )
        goto LABEL_28;
    }
    goto LABEL_22;
  }
LABEL_28:
  if ( (unsigned int)v13 < *(_DWORD *)(a1 + 16) )
  {
    v13 = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 20) = v13;
    *(_QWORD *)v19 = v23;
    v22 = *(_BYTE *)(a1 + 28);
    ++*(_QWORD *)a1;
LABEL_22:
    if ( v24 != ++v21 )
      goto LABEL_23;
    return result;
  }
LABEL_24:
  while ( 1 )
  {
    ++v21;
    result = (unsigned __int64)sub_C8CC70(a1, v23, v19, v13, v14, v15);
    v22 = *(_BYTE *)(a1 + 28);
    if ( v24 == v21 )
      return result;
LABEL_23:
    v23 = *v21;
    if ( v22 )
      goto LABEL_18;
  }
}
