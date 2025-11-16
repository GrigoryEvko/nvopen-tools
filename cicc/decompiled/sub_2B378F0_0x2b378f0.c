// Function: sub_2B378F0
// Address: 0x2b378f0
//
__int64 *__fastcall sub_2B378F0(__int64 a1, __int64 a2, char *a3, __int64 a4)
{
  __int64 v4; // r9
  unsigned __int64 v5; // r8
  __int64 v9; // rdi
  __int64 v11; // r15
  int v12; // eax
  int v13; // esi
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // edx
  unsigned int v17; // eax
  unsigned int v18; // ecx
  char *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *result; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  int v26; // edi
  __int64 v27; // rbx
  __int64 v28; // r9
  __int64 v29; // rbx
  __int64 *v30; // [rsp-10h] [rbp-50h]
  __int64 v31; // [rsp+0h] [rbp-40h]
  __int64 v32; // [rsp+0h] [rbp-40h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  unsigned __int64 v34; // [rsp+8h] [rbp-38h]
  unsigned __int64 v35; // [rsp+8h] [rbp-38h]

  v4 = 4 * a4;
  v5 = (4 * a4) >> 2;
  if ( !*(_DWORD *)(a1 + 88) )
  {
    v23 = *(unsigned int *)(a1 + 28);
    *(_DWORD *)(a1 + 24) = 0;
    LODWORD(v24) = 0;
    v25 = 0;
    if ( v5 > v23 )
    {
      v32 = 4 * a4;
      v35 = (4 * a4) >> 2;
      sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v35, 4u, v5, v4);
      v24 = *(unsigned int *)(a1 + 24);
      v4 = v32;
      v5 = v35;
      v25 = 4 * v24;
    }
    if ( v4 )
    {
      v34 = v5;
      memcpy((void *)(*(_QWORD *)(a1 + 16) + v25), a3, v4);
      LODWORD(v24) = *(_DWORD *)(a1 + 24);
      v5 = v34;
    }
    v26 = *(_DWORD *)(a1 + 92);
    v27 = a2 | 4;
    *(_DWORD *)(a1 + 24) = v5 + v24;
    if ( v26 )
    {
      if ( *(_DWORD *)(a1 + 88) )
      {
        result = *(__int64 **)(a1 + 80);
        *result = v27;
        if ( *(_DWORD *)(a1 + 88) )
        {
LABEL_31:
          *(_DWORD *)(a1 + 88) = 1;
          return result;
        }
      }
    }
    else
    {
      *(_DWORD *)(a1 + 88) = 0;
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), 1u, 8u, v5, v4);
    }
    result = *(__int64 **)(a1 + 80);
    if ( result )
      *result = v27;
    goto LABEL_31;
  }
  v9 = *(_QWORD *)a1;
  v11 = (4 * a4) >> 2;
  v12 = *(unsigned __int8 *)(v9 + 8);
  if ( (_BYTE)v12 == 17 )
  {
    v13 = a4 * *(_DWORD *)(v9 + 32);
LABEL_4:
    v9 = **(_QWORD **)(v9 + 16);
    goto LABEL_5;
  }
  v13 = a4;
  if ( (unsigned int)(v12 - 17) <= 1 )
    goto LABEL_4;
LABEL_5:
  v31 = (4 * a4) >> 2;
  v33 = 4 * a4;
  v14 = sub_BCDA70((__int64 *)v9, v13);
  v15 = sub_2B1F810(*(_QWORD *)(a1 + 112), v14, a4);
  v16 = 1;
  v17 = ((_DWORD)a4 != 0) + ((unsigned int)a4 - ((_DWORD)a4 != 0)) / v15;
  if ( v17 > 1 )
  {
    _BitScanReverse(&v17, v17 - 1);
    v16 = 1 << (32 - (v17 ^ 0x1F));
  }
  v18 = v16;
  if ( (unsigned int)a4 <= v16 )
    v18 = a4;
  if ( v31 >> 2 > 0 )
  {
    v19 = a3;
    while ( *(_DWORD *)v19 == -1 )
    {
      if ( *((_DWORD *)v19 + 1) != -1 )
      {
        v11 = (v19 + 4 - a3) >> 2;
        goto LABEL_17;
      }
      if ( *((_DWORD *)v19 + 2) != -1 )
      {
        v11 = (v19 + 8 - a3) >> 2;
        goto LABEL_17;
      }
      if ( *((_DWORD *)v19 + 3) != -1 )
      {
        v11 = (v19 + 12 - a3) >> 2;
        goto LABEL_17;
      }
      v19 += 16;
      if ( &a3[16 * (v31 >> 2)] == v19 )
      {
        v28 = (&a3[v33] - v19) >> 2;
        goto LABEL_36;
      }
    }
    goto LABEL_16;
  }
  v28 = v31;
  v19 = a3;
LABEL_36:
  if ( v28 == 2 )
  {
LABEL_47:
    if ( *(_DWORD *)v19 == -1 )
    {
      v19 += 4;
LABEL_39:
      if ( *(_DWORD *)v19 == -1 )
        goto LABEL_17;
      goto LABEL_16;
    }
    goto LABEL_16;
  }
  if ( v28 != 3 )
  {
    if ( v28 != 1 )
      goto LABEL_17;
    goto LABEL_39;
  }
  if ( *(_DWORD *)v19 == -1 )
  {
    v19 += 4;
    goto LABEL_47;
  }
LABEL_16:
  v11 = (v19 - a3) >> 2;
LABEL_17:
  sub_2B373F0(a1, a2, 0, a3, a4, v11 / v18, v18);
  result = v30;
  if ( !*(_BYTE *)(a1 + 200) && *(_DWORD *)(a1 + 88) == 1 )
  {
    v29 = a2 | 4;
    if ( *(_DWORD *)(a1 + 92) <= 1u )
    {
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), 2u, 8u, v20, v21);
      result = *(__int64 **)(a1 + 80);
      result[*(unsigned int *)(a1 + 88)] = v29;
    }
    else
    {
      result = *(__int64 **)(a1 + 80);
      result[1] = v29;
    }
    ++*(_DWORD *)(a1 + 88);
  }
  return result;
}
