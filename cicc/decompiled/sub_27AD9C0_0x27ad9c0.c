// Function: sub_27AD9C0
// Address: 0x27ad9c0
//
__int64 __fastcall sub_27AD9C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 i; // rbx
  int v7; // esi
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // r11
  __int64 v12; // r10
  int v13; // esi
  unsigned int v14; // r9d
  __int64 *v15; // rdx
  __int64 v16; // r15
  unsigned int v17; // r15d
  unsigned int v18; // r9d
  __int64 *v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r15
  __int64 v22; // r10
  unsigned int v23; // r14d
  int v24; // esi
  unsigned int v25; // ebx
  __int64 *v26; // rdx
  __int64 v27; // r12
  unsigned int v28; // ebx
  unsigned int v29; // r13d
  __int64 *v30; // rdx
  __int64 v31; // r12
  __int64 *v32; // rax
  int v33; // esi
  __int64 v34; // r11
  __int64 v35; // r9
  int v37; // edx
  int v38; // edx
  int v39; // edx
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // r13d
  int v44; // r13d
  int v45; // [rsp+0h] [rbp-4Ch]
  __int64 v47; // [rsp+Ch] [rbp-40h]
  int v49; // [rsp+14h] [rbp-38h]

  v5 = (a3 - 1) / 2;
  v47 = a3 & 1;
  if ( a2 >= v5 )
  {
    v9 = a2;
    v10 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_22;
  }
  else
  {
    for ( i = a2; ; i = v9 )
    {
      v7 = *(_DWORD *)(a5 + 592);
      v8 = *(_QWORD *)(a5 + 576);
      v9 = 2 * (i + 1);
      v10 = (__int64 *)(a1 + 16 * (i + 1));
      v11 = *(v10 - 1);
      v12 = *v10;
      if ( v7 )
      {
        v13 = v7 - 1;
        v14 = v13 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v15 = (__int64 *)(v8 + 16LL * v14);
        v16 = *v15;
        if ( v12 == *v15 )
        {
LABEL_6:
          v17 = *((_DWORD *)v15 + 2);
        }
        else
        {
          v40 = 1;
          while ( v16 != -4096 )
          {
            v44 = v40 + 1;
            v14 = v13 & (v40 + v14);
            v15 = (__int64 *)(v8 + 16LL * v14);
            v16 = *v15;
            if ( v12 == *v15 )
              goto LABEL_6;
            v40 = v44;
          }
          v17 = 0;
        }
        v18 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v19 = (__int64 *)(v8 + 16LL * v18);
        v20 = *v19;
        if ( v11 == *v19 )
        {
LABEL_8:
          if ( *((_DWORD *)v19 + 2) > v17 )
          {
            --v9;
            v10 = (__int64 *)(a1 + 8 * v9);
            v12 = *v10;
          }
        }
        else
        {
          v39 = 1;
          while ( v20 != -4096 )
          {
            v18 = v13 & (v39 + v18);
            v45 = v39 + 1;
            v19 = (__int64 *)(v8 + 16LL * v18);
            v20 = *v19;
            if ( v11 == *v19 )
              goto LABEL_8;
            v39 = v45;
          }
        }
      }
      *(_QWORD *)(a1 + 8 * i) = v12;
      if ( v9 >= v5 )
        break;
    }
    if ( v47 )
      goto LABEL_12;
  }
  if ( (a3 - 2) / 2 == v9 )
  {
    v41 = 2 * v9 + 2;
    v42 = *(_QWORD *)(a1 + 8 * v41 - 8);
    v9 = v41 - 1;
    *v10 = v42;
    v10 = (__int64 *)(a1 + 8 * v9);
  }
LABEL_12:
  v21 = a2;
  v22 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    v23 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
    while ( 1 )
    {
      v33 = *(_DWORD *)(a5 + 592);
      v10 = (__int64 *)(a1 + 8 * v22);
      v34 = *(_QWORD *)(a5 + 576);
      v35 = *v10;
      if ( !v33 )
        goto LABEL_21;
      v24 = v33 - 1;
      v25 = v24 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v26 = (__int64 *)(v34 + 16LL * v25);
      v27 = *v26;
      if ( v35 == *v26 )
      {
LABEL_15:
        v28 = *((_DWORD *)v26 + 2);
      }
      else
      {
        v38 = 1;
        while ( v27 != -4096 )
        {
          v43 = v38 + 1;
          v25 = v24 & (v38 + v25);
          v26 = (__int64 *)(v34 + 16LL * v25);
          v27 = *v26;
          if ( v35 == *v26 )
            goto LABEL_15;
          v38 = v43;
        }
        v28 = 0;
      }
      v29 = v23 & v24;
      v30 = (__int64 *)(v34 + 16LL * (v23 & v24));
      v31 = *v30;
      if ( a4 != *v30 )
      {
        v37 = 1;
        while ( v31 != -4096 )
        {
          v29 = v24 & (v37 + v29);
          v49 = v37 + 1;
          v30 = (__int64 *)(v34 + 16LL * v29);
          v31 = *v30;
          if ( a4 == *v30 )
            goto LABEL_17;
          v37 = v49;
        }
LABEL_21:
        v10 = (__int64 *)(a1 + 8 * v9);
        goto LABEL_22;
      }
LABEL_17:
      v32 = (__int64 *)(a1 + 8 * v9);
      if ( *((_DWORD *)v30 + 2) <= v28 )
        break;
      *v32 = v35;
      v9 = v22;
      if ( v21 >= v22 )
        goto LABEL_22;
      v22 = (v22 - 1) / 2;
    }
    v10 = v32;
  }
LABEL_22:
  *v10 = a4;
  return a4;
}
