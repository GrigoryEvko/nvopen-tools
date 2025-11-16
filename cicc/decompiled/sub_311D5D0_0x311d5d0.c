// Function: sub_311D5D0
// Address: 0x311d5d0
//
unsigned int *__fastcall sub_311D5D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  unsigned int v6; // r14d
  unsigned int v8; // r12d
  __int64 v9; // rbx
  unsigned int v10; // r15d
  __int64 i; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned int *result; // rax
  unsigned int v16; // esi
  unsigned int v17; // r10d
  __int64 v18; // rcx
  __int64 v19; // rcx
  unsigned int v20; // esi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // [rsp+8h] [rbp-38h]
  unsigned __int64 v25; // [rsp+10h] [rbp-30h]

  v6 = a4;
  v8 = a4;
  v9 = (a3 - 1) / 2;
  v24 = a3 & 1;
  v25 = HIDWORD(a4);
  v10 = HIDWORD(a4);
  if ( a2 >= v9 )
  {
    result = (unsigned int *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_27;
    v12 = a2;
    goto LABEL_24;
  }
  for ( i = a2; ; i = v12 )
  {
    v12 = 2 * (i + 1);
    v13 = 32 * (i + 1);
    v14 = a1 + v13 - 16;
    result = (unsigned int *)(a1 + v13);
    v16 = *result;
    v17 = *(_DWORD *)v14;
    if ( *result < *(_DWORD *)v14
      || v16 == v17 && result[1] < *(_DWORD *)(v14 + 4)
      || v16 <= v17 && *(_DWORD *)(v14 + 4) >= result[1] && *((_QWORD *)result + 1) < *(_QWORD *)(v14 + 8) )
    {
      --v12;
      result = (unsigned int *)(a1 + 16 * v12);
      v16 = *result;
    }
    v18 = a1 + 16 * i;
    *(_DWORD *)v18 = v16;
    *(_DWORD *)(v18 + 4) = result[1];
    *(_QWORD *)(v18 + 8) = *((_QWORD *)result + 1);
    if ( v12 >= v9 )
      break;
  }
  if ( !v24 )
  {
LABEL_24:
    if ( (a3 - 2) / 2 == v12 )
    {
      v22 = v12 + 1;
      v12 = 2 * (v12 + 1) - 1;
      v23 = a1 + 32 * v22 - 16;
      *result = *(_DWORD *)v23;
      result[1] = *(_DWORD *)(v23 + 4);
      *((_QWORD *)result + 1) = *(_QWORD *)(v23 + 8);
      result = (unsigned int *)(a1 + 16 * v12);
    }
  }
  v19 = (v12 - 1) / 2;
  if ( v12 > a2 )
  {
    while ( 1 )
    {
      result = (unsigned int *)(a1 + 16 * v19);
      v20 = *result;
      if ( *result >= v8
        && (v20 != v8 || result[1] >= v10)
        && (v20 > v8 || result[1] > v10 || *((_QWORD *)result + 1) >= a5) )
      {
        break;
      }
      v21 = a1 + 16 * v12;
      *(_DWORD *)v21 = v20;
      *(_DWORD *)(v21 + 4) = result[1];
      *(_QWORD *)(v21 + 8) = *((_QWORD *)result + 1);
      v12 = v19;
      if ( a2 >= v19 )
        goto LABEL_27;
      v19 = (v19 - 1) / 2;
    }
    result = (unsigned int *)(a1 + 16 * v12);
  }
LABEL_27:
  *result = v6;
  *((_QWORD *)result + 1) = a5;
  result[1] = v25;
  return result;
}
