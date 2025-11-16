// Function: sub_3925090
// Address: 0x3925090
//
signed __int64 __fastcall sub_3925090(char *a1, char *a2, __int64 a3)
{
  signed __int64 result; // rax
  char *v5; // r14
  __int64 v6; // rbx
  char *v7; // r10
  char *v8; // r12
  __int64 v9; // r13
  __int64 v10; // r8
  int v11; // esi
  char *v12; // rdi
  __int64 v13; // rax
  int v14; // ecx
  int v15; // edx
  __int64 v16; // rdx
  int v17; // esi
  __int64 *v18; // rdi
  char *v19; // r13
  char *v20; // rcx
  char *v21; // rax
  __int64 v22; // rbx
  __int64 i; // rsi
  __int64 *v24; // r14
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 *v27; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( a2 - a1 <= 128 )
    return result;
  v5 = a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 8;
  v27 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v9 = *((_QWORD *)v7 - 1);
    v10 = *(_QWORD *)a1;
    --v6;
    v11 = *(_DWORD *)(v9 + 72);
    v12 = &a1[8 * ((__int64)(((v7 - a1) >> 3) + ((unsigned __int64)(v7 - a1) >> 63)) >> 1)];
    v13 = *((_QWORD *)a1 + 1);
    v14 = *(_DWORD *)(v13 + 72);
    v15 = *(_DWORD *)(*(_QWORD *)v12 + 72LL);
    if ( v14 >= v15 )
    {
      if ( v14 < v11 )
        goto LABEL_7;
      if ( v15 < v11 )
      {
LABEL_18:
        *(_QWORD *)a1 = v9;
        v16 = v10;
        *((_QWORD *)v7 - 1) = v10;
        v13 = *(_QWORD *)a1;
        v10 = *((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_23:
      *(_QWORD *)a1 = *(_QWORD *)v12;
      *(_QWORD *)v12 = v10;
      v13 = *(_QWORD *)a1;
      v10 = *((_QWORD *)a1 + 1);
      v16 = *((_QWORD *)v7 - 1);
      goto LABEL_8;
    }
    if ( v15 < v11 )
      goto LABEL_23;
    if ( v14 < v11 )
      goto LABEL_18;
LABEL_7:
    *(_QWORD *)a1 = v13;
    *((_QWORD *)a1 + 1) = v10;
    v16 = *((_QWORD *)v7 - 1);
LABEL_8:
    v17 = *(_DWORD *)(v13 + 72);
    v18 = v27;
    v19 = v8;
    v20 = v7;
    while ( 1 )
    {
      v5 = v19;
      if ( *(_DWORD *)(v10 + 72) < v17 )
        goto LABEL_15;
      if ( v17 >= *(_DWORD *)(v16 + 72) )
      {
        v20 -= 8;
      }
      else
      {
        v21 = v20 - 16;
        do
        {
          v16 = *(_QWORD *)v21;
          v20 = v21;
          v21 -= 8;
        }
        while ( *(_DWORD *)(v16 + 72) > v17 );
      }
      if ( v19 >= v20 )
        break;
      *(_QWORD *)v19 = v16;
      v16 = *((_QWORD *)v20 - 1);
      *(_QWORD *)v20 = v10;
      v17 = *(_DWORD *)(*(_QWORD *)a1 + 72LL);
LABEL_15:
      v10 = *v18;
      v19 += 8;
      ++v18;
    }
    sub_3925090(v19, v7, v6);
    result = v19 - a1;
    if ( v19 - a1 > 128 )
    {
      if ( v6 )
      {
        v7 = v19;
        continue;
      }
LABEL_24:
      v22 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_3924E80((__int64)a1, i, v22, *(_QWORD *)&a1[8 * i]);
        if ( !i )
          break;
      }
      v24 = (__int64 *)(v5 - 8);
      do
      {
        v25 = *v24;
        v26 = (char *)v24-- - a1;
        v24[1] = *(_QWORD *)a1;
        result = (signed __int64)sub_3924E80((__int64)a1, 0, v26 >> 3, v25);
      }
      while ( v26 > 8 );
    }
    return result;
  }
}
