// Function: sub_E025C0
// Address: 0xe025c0
//
_QWORD *__fastcall sub_E025C0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r10
  _QWORD *result; // rax
  __int64 v9; // r14
  __int64 v10; // rbx
  const void *i; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rdx
  const void *v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  _QWORD *v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  const void *v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]
  __int64 v31; // [rsp+28h] [rbp-38h]
  __int64 v32; // [rsp+28h] [rbp-38h]

  v6 = a3;
  result = (_QWORD *)(32 * (1LL - (*(_DWORD *)(a5 + 4) & 0x7FFFFFF)));
  v9 = *(_QWORD *)((char *)result + a5);
  if ( *(_BYTE *)v9 != 17 )
  {
    *a4 = 1;
    return result;
  }
  v10 = *(_QWORD *)(a5 + 16);
  for ( i = (const void *)(a3 + 16); v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    v17 = *(_QWORD *)(v10 + 24);
    if ( *(_BYTE *)v17 != 93 || *(_DWORD *)(v17 + 80) != 1 )
      goto LABEL_9;
    v18 = **(_DWORD **)(v17 + 72);
    if ( !v18 )
    {
      v20 = *(unsigned int *)(a2 + 8);
      if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v21 = i;
        v23 = a5;
        v25 = v6;
        v28 = *(_QWORD *)(v10 + 24);
        v31 = a2;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v20 + 1, 8u, a5, v17);
        a2 = v31;
        i = v21;
        a5 = v23;
        v6 = v25;
        v20 = *(unsigned int *)(v31 + 8);
        v17 = v28;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v20) = v17;
      ++*(_DWORD *)(a2 + 8);
      continue;
    }
    if ( v18 == 1 )
    {
      v19 = *(unsigned int *)(v6 + 8);
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
      {
        v22 = a5;
        v24 = a2;
        v26 = *(_QWORD *)(v10 + 24);
        v29 = i;
        v32 = v6;
        sub_C8D5F0(v6, i, v19 + 1, 8u, a5, v17);
        v6 = v32;
        a5 = v22;
        a2 = v24;
        v17 = v26;
        v19 = *(unsigned int *)(v32 + 8);
        i = v29;
      }
      *(_QWORD *)(*(_QWORD *)v6 + 8 * v19) = v17;
      ++*(_DWORD *)(v6 + 8);
    }
    else
    {
LABEL_9:
      *a4 = 1;
    }
  }
  v14 = *(_QWORD **)a2;
  result = (_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
  v27 = result;
  if ( result != *(_QWORD **)a2 )
  {
    do
    {
      v15 = *(_QWORD **)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) > 0x40u )
        v15 = (_QWORD *)*v15;
      v16 = *(_QWORD *)(*v14 + 16LL);
      v30 = a5;
      ++v14;
      result = sub_E02020(a1, a4, v16, (__int64)v15, a5, a6);
      a5 = v30;
    }
    while ( v27 != v14 );
  }
  return result;
}
