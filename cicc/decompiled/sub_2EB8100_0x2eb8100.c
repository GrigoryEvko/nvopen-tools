// Function: sub_2EB8100
// Address: 0x2eb8100
//
__int64 __fastcall sub_2EB8100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 i; // r13
  __int64 v8; // r15
  __int64 *v9; // r14
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // r13
  unsigned int v13; // r10d
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 *v16; // rdx
  __int64 *v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int v20; // edi
  __int64 v21; // rax
  __int64 v23; // rdx
  int v24; // eax
  int v25; // ecx
  __int64 v27; // [rsp+10h] [rbp-B0h]
  __int64 v28; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+38h] [rbp-88h]
  unsigned int v32; // [rsp+44h] [rbp-7Ch]
  __int64 *v33[4]; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v34[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v35; // [rsp+80h] [rbp-40h]

  v5 = a1;
  v28 = (a3 - 1) / 2;
  v27 = a3 & 1;
  if ( a2 >= v28 )
  {
    v8 = a2;
    v9 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_17;
    goto LABEL_19;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = (__int64 *)(a1 + 16 * (i + 1));
    v31 = *(v9 - 1);
    sub_2E6E850(v33, (__int64 *)a5, *v9);
    v32 = *((_DWORD *)v33[2] + 2);
    sub_2E6E850(v34, (__int64 *)a5, v31);
    if ( v32 < *(_DWORD *)(v35 + 8) )
    {
      --v8;
      v9 = (__int64 *)(a1 + 8 * v8);
    }
    *(_QWORD *)(a1 + 8 * i) = *v9;
    if ( v8 >= v28 )
      break;
  }
  v5 = a1;
  if ( !v27 )
  {
LABEL_19:
    if ( (a3 - 2) / 2 == v8 )
    {
      v23 = *(_QWORD *)(v5 + 8 * (2 * v8 + 2) - 8);
      v8 = 2 * v8 + 1;
      *v9 = v23;
      v9 = (__int64 *)(v5 + 8 * v8);
    }
  }
  v10 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
    v11 = v8;
    v12 = v5;
    while ( 1 )
    {
      v17 = (__int64 *)(v12 + 8 * v10);
      sub_2E6E850(v34, (__int64 *)a5, *v17);
      v18 = *(unsigned int *)(a5 + 24);
      v19 = *(_QWORD *)(a5 + 8);
      v20 = *(_DWORD *)(v35 + 8);
      if ( (_DWORD)v18 )
      {
        v13 = (v18 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
        v14 = (__int64 *)(v19 + 16LL * v13);
        v15 = *v14;
        if ( a4 == *v14 )
        {
LABEL_11:
          v16 = (__int64 *)(v12 + 8 * v11);
          if ( v20 >= *((_DWORD *)v14 + 2) )
            goto LABEL_16;
          goto LABEL_12;
        }
        v24 = 1;
        while ( v15 != -4096 )
        {
          v25 = v24 + 1;
          v13 = (v18 - 1) & (v24 + v13);
          v14 = (__int64 *)(v19 + 16LL * v13);
          v15 = *v14;
          if ( a4 == *v14 )
            goto LABEL_11;
          v24 = v25;
        }
      }
      v21 = v19 + 16 * v18;
      v16 = (__int64 *)(v12 + 8 * v11);
      if ( v20 >= *(_DWORD *)(v21 + 8) )
      {
LABEL_16:
        v9 = v16;
        break;
      }
LABEL_12:
      v11 = v10;
      *v16 = *v17;
      if ( a2 >= v10 )
      {
        v9 = (__int64 *)(v12 + 8 * v10);
        break;
      }
      v10 = (v10 - 1) / 2;
    }
  }
LABEL_17:
  *v9 = a4;
  return a4;
}
