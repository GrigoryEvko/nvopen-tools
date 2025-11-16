// Function: sub_80B480
// Address: 0x80b480
//
__int64 __fastcall sub_80B480(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rbx
  __int64 j; // rbx
  char *v5; // r15
  size_t v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // rdx
  __int64 k; // rbx
  char *v14; // r14
  size_t v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int8 *v20; // rax
  unsigned __int8 *v21; // rdx
  size_t v22; // [rsp+0h] [rbp-80h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  char v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  int v27; // [rsp+38h] [rbp-48h]
  char v28; // [rsp+3Ch] [rbp-44h]
  __int64 v29; // [rsp+40h] [rbp-40h]

  result = sub_80B6E0(a1[13]);
  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      result = sub_80B480(*(_QWORD *)(i + 128));
  }
  for ( j = a1[18]; j; j = *(_QWORD *)(j + 112) )
  {
    while ( (*(_DWORD *)(j + 192) & 0x8000400) != 0 || (*(_BYTE *)(j + 89) & 0x20) == 0 )
    {
      j = *(_QWORD *)(j + 112);
      if ( !j )
        goto LABEL_13;
    }
    v5 = *(char **)(j + 8);
    v6 = strlen(v5) + 1;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    sub_809110(v5, a2, v7, v8, v9, v10, 0, 0, 0);
    sub_823800(qword_4F18BE0);
    v22 = v6;
    v11 = sub_809930((unsigned __int8 *)v5, j, (__int64)&v22);
    a2 = qword_4F18BE8;
    v12 = v11;
    result = *(_QWORD *)qword_4F18BF0;
    qword_4F18BE8 = qword_4F18BF0;
    *(_QWORD *)qword_4F18BF0 = a2;
    qword_4F18BF0 = result;
    if ( result )
      result = *(_QWORD *)(result + 8);
    qword_4F18BE0 = result;
    *(_QWORD *)(j + 8) = v12;
    *(_BYTE *)(j + 89) &= ~0x20u;
  }
LABEL_13:
  for ( k = a1[14]; k; k = *(_QWORD *)(k + 112) )
  {
    while ( (*(_BYTE *)(k + 89) & 0x20) == 0 )
    {
      k = *(_QWORD *)(k + 112);
      if ( !k )
        return result;
    }
    v14 = *(char **)(k + 8);
    v15 = strlen(v14) + 1;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    sub_809110(v14, a2, v16, v17, v18, v19, 0, 0, 0);
    sub_823800(qword_4F18BE0);
    v22 = v15;
    v20 = sub_809930((unsigned __int8 *)v14, k, (__int64)&v22);
    a2 = qword_4F18BE8;
    v21 = v20;
    result = *(_QWORD *)qword_4F18BF0;
    qword_4F18BE8 = qword_4F18BF0;
    *(_QWORD *)qword_4F18BF0 = a2;
    qword_4F18BF0 = result;
    if ( result )
      result = *(_QWORD *)(result + 8);
    qword_4F18BE0 = result;
    *(_QWORD *)(k + 8) = v21;
    *(_BYTE *)(k + 89) &= ~0x20u;
  }
  return result;
}
