// Function: sub_2B37C80
// Address: 0x2b37c80
//
__int64 *__fastcall sub_2B37C80(__int64 a1, __int64 a2, __int64 a3, char *a4, __int64 a5)
{
  size_t v9; // r10
  unsigned __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  int v14; // eax
  int v15; // esi
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // r9
  unsigned int v19; // edx
  unsigned int v20; // eax
  unsigned int v21; // ecx
  char *v22; // rdx
  __int64 *result; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 v29; // r10
  __int64 v30; // [rsp-10h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  size_t v32; // [rsp+10h] [rbp-40h]
  size_t v33; // [rsp+10h] [rbp-40h]
  unsigned __int64 v34; // [rsp+18h] [rbp-38h]
  unsigned __int64 v35; // [rsp+18h] [rbp-38h]
  unsigned __int64 v36; // [rsp+18h] [rbp-38h]

  if ( a2 == a3 )
    return sub_2B378F0(a1, a2, a4, a5);
  v9 = 4 * a5;
  v11 = (4 * a5) >> 2;
  v12 = v11;
  if ( *(_DWORD *)(a1 + 88) )
  {
    v13 = *(_QWORD *)a1;
    v14 = *(unsigned __int8 *)(v13 + 8);
    if ( (_BYTE)v14 == 17 )
    {
      v15 = a5 * *(_DWORD *)(v13 + 32);
    }
    else
    {
      v15 = a5;
      if ( (unsigned int)(v14 - 17) > 1 )
      {
LABEL_6:
        v31 = v11;
        v32 = v9;
        v34 = v11;
        v16 = sub_BCDA70((__int64 *)v13, v15);
        v17 = sub_2B1F810(*(_QWORD *)(a1 + 112), v16, a5);
        v18 = v34;
        v19 = 1;
        v20 = ((_DWORD)a5 != 0) + ((unsigned int)a5 - ((_DWORD)a5 != 0)) / v17;
        if ( v20 > 1 )
        {
          _BitScanReverse(&v20, v20 - 1);
          v19 = 1 << (32 - (v20 ^ 0x1F));
        }
        v21 = v19;
        if ( (unsigned int)a5 <= v19 )
          v21 = a5;
        if ( v31 >> 2 > 0 )
        {
          v22 = a4;
          while ( *(_DWORD *)v22 == -1 )
          {
            if ( *((_DWORD *)v22 + 1) != -1 )
            {
              v18 = (v22 + 4 - a4) >> 2;
              goto LABEL_18;
            }
            if ( *((_DWORD *)v22 + 2) != -1 )
            {
              v18 = (v22 + 8 - a4) >> 2;
              goto LABEL_18;
            }
            if ( *((_DWORD *)v22 + 3) != -1 )
            {
              v18 = (v22 + 12 - a4) >> 2;
              goto LABEL_18;
            }
            v22 += 16;
            if ( &a4[16 * (v31 >> 2)] == v22 )
            {
              v29 = (&a4[v32] - v22) >> 2;
              goto LABEL_32;
            }
          }
          goto LABEL_17;
        }
        v29 = v31;
        v22 = a4;
LABEL_32:
        if ( v29 != 2 )
        {
          if ( v29 != 3 )
          {
            if ( v29 != 1 )
              goto LABEL_18;
            goto LABEL_35;
          }
          if ( *(_DWORD *)v22 != -1 )
          {
LABEL_17:
            v18 = (v22 - a4) >> 2;
LABEL_18:
            sub_2B373F0(a1, a2, a3, a4, a5, v18 / v21, v21);
            return (__int64 *)v30;
          }
          v22 += 4;
        }
        if ( *(_DWORD *)v22 == -1 )
        {
          v22 += 4;
LABEL_35:
          if ( *(_DWORD *)v22 == -1 )
            goto LABEL_18;
          goto LABEL_17;
        }
        goto LABEL_17;
      }
    }
    v13 = **(_QWORD **)(v13 + 16);
    goto LABEL_6;
  }
  v24 = *(unsigned int *)(a1 + 28);
  *(_DWORD *)(a1 + 24) = 0;
  LODWORD(v25) = 0;
  v26 = 0;
  if ( v11 > v24 )
  {
    v33 = v9;
    v36 = v11;
    sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v11, 4u, v11, v11);
    v25 = *(unsigned int *)(a1 + 24);
    v9 = v33;
    v11 = v36;
    v26 = 4 * v25;
  }
  if ( v9 )
  {
    v35 = v11;
    memcpy((void *)(*(_QWORD *)(a1 + 16) + v26), a4, v9);
    LODWORD(v25) = *(_DWORD *)(a1 + 24);
    v11 = v35;
  }
  v27 = a3 | 4;
  *(_DWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 24) = v11 + v25;
  result = 0;
  if ( *(_DWORD *)(a1 + 92) <= 1u )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), 2u, 8u, v11, v12);
    result = (__int64 *)(8LL * *(unsigned int *)(a1 + 88));
  }
  v28 = *(_QWORD *)(a1 + 80);
  *(__int64 *)((char *)result + v28) = a2 | 4;
  *(__int64 *)((char *)result + v28 + 8) = v27;
  *(_DWORD *)(a1 + 88) += 2;
  return result;
}
