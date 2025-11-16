// Function: sub_37B82D0
// Address: 0x37b82d0
//
__int64 __fastcall sub_37B82D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // rbx
  __int64 v6; // r10
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 *v9; // rcx
  __int64 v10; // r11
  __int64 v11; // r9
  int v12; // r13d
  unsigned int v13; // esi
  __int64 *v14; // rdx
  __int64 v15; // r14
  unsigned int v16; // r15d
  unsigned int v17; // esi
  __int64 *v18; // rdx
  __int64 v19; // r14
  unsigned int v20; // edx
  __int64 v21; // r9
  unsigned int v22; // r15d
  int v23; // ebx
  unsigned int v24; // r12d
  __int64 *v25; // rdx
  __int64 v26; // r13
  unsigned int v27; // r12d
  unsigned int v28; // r14d
  __int64 *v29; // rdx
  __int64 v30; // r13
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // r10
  __int64 v34; // r11
  __int64 v35; // rsi
  int v37; // edx
  int v38; // edx
  int v39; // edx
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // r14d
  int v44; // r15d
  int v45; // [rsp+0h] [rbp-54h]
  __int64 v47; // [rsp+Ch] [rbp-48h]
  __int64 v48; // [rsp+14h] [rbp-40h]
  int v49; // [rsp+14h] [rbp-40h]

  v47 = a3 & 1;
  v48 = (a3 - 1) / 2;
  if ( a2 >= v48 )
  {
    v8 = a2;
    v9 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_24;
  }
  else
  {
    for ( i = a2; ; i = v8 )
    {
      v6 = *(unsigned int *)(a5 + 688);
      v7 = *(_QWORD *)(a5 + 672);
      v8 = 2 * (i + 1);
      v9 = (__int64 *)(a1 + 16 * (i + 1));
      v10 = *(v9 - 1);
      v11 = *v9;
      if ( (_DWORD)v6 )
      {
        v12 = v6 - 1;
        v13 = (v6 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v14 = (__int64 *)(v7 + 16LL * v13);
        v15 = *v14;
        if ( v11 == *v14 )
        {
LABEL_6:
          v16 = *((_DWORD *)v14 + 2);
        }
        else
        {
          v40 = 1;
          while ( v15 != -4096 )
          {
            v44 = v40 + 1;
            v13 = v12 & (v40 + v13);
            v14 = (__int64 *)(v7 + 16LL * v13);
            v15 = *v14;
            if ( v11 == *v14 )
              goto LABEL_6;
            v40 = v44;
          }
          v16 = *(_DWORD *)(v7 + 16LL * (unsigned int)v6 + 8);
        }
        v17 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v18 = (__int64 *)(v7 + 16LL * v17);
        v19 = *v18;
        if ( v10 == *v18 )
        {
LABEL_8:
          v20 = *((_DWORD *)v18 + 2);
        }
        else
        {
          v39 = 1;
          while ( v19 != -4096 )
          {
            v17 = v12 & (v39 + v17);
            v45 = v39 + 1;
            v18 = (__int64 *)(v7 + 16LL * v17);
            v19 = *v18;
            if ( v10 == *v18 )
              goto LABEL_8;
            v39 = v45;
          }
          v20 = *(_DWORD *)(v7 + 16 * v6 + 8);
        }
        if ( v20 > v16 )
        {
          --v8;
          v9 = (__int64 *)(a1 + 8 * v8);
          v11 = *v9;
        }
      }
      *(_QWORD *)(a1 + 8 * i) = v11;
      if ( v8 >= v48 )
        break;
    }
    if ( v47 )
      goto LABEL_13;
  }
  if ( (a3 - 2) / 2 == v8 )
  {
    v41 = 2 * v8 + 2;
    v42 = *(_QWORD *)(a1 + 8 * v41 - 8);
    v8 = v41 - 1;
    *v9 = v42;
    v9 = (__int64 *)(a1 + 8 * v8);
  }
LABEL_13:
  v21 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
    v22 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
    while ( 1 )
    {
      v33 = *(unsigned int *)(a5 + 688);
      v9 = (__int64 *)(a1 + 8 * v21);
      v34 = *(_QWORD *)(a5 + 672);
      v35 = *v9;
      if ( !(_DWORD)v33 )
      {
        v9 = (__int64 *)(a1 + 8 * v8);
        goto LABEL_24;
      }
      v23 = v33 - 1;
      v24 = (v33 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v25 = (__int64 *)(v34 + 16LL * v24);
      v26 = *v25;
      if ( v35 == *v25 )
      {
LABEL_16:
        v27 = *((_DWORD *)v25 + 2);
      }
      else
      {
        v38 = 1;
        while ( v26 != -4096 )
        {
          v43 = v38 + 1;
          v24 = v23 & (v38 + v24);
          v25 = (__int64 *)(v34 + 16LL * v24);
          v26 = *v25;
          if ( v35 == *v25 )
            goto LABEL_16;
          v38 = v43;
        }
        v27 = *(_DWORD *)(v34 + 16LL * (unsigned int)v33 + 8);
      }
      v28 = v22 & v23;
      v29 = (__int64 *)(v34 + 16LL * (v22 & v23));
      v30 = *v29;
      if ( a4 == *v29 )
      {
LABEL_18:
        v31 = *((_DWORD *)v29 + 2);
      }
      else
      {
        v37 = 1;
        while ( v30 != -4096 )
        {
          v28 = v23 & (v37 + v28);
          v49 = v37 + 1;
          v29 = (__int64 *)(v34 + 16LL * v28);
          v30 = *v29;
          if ( a4 == *v29 )
            goto LABEL_18;
          v37 = v49;
        }
        v31 = *(_DWORD *)(v34 + 16 * v33 + 8);
      }
      v32 = (__int64 *)(a1 + 8 * v8);
      if ( v31 <= v27 )
        break;
      *v32 = v35;
      v8 = v21;
      if ( a2 >= v21 )
        goto LABEL_24;
      v21 = (v21 - 1) / 2;
    }
    v9 = v32;
  }
LABEL_24:
  *v9 = a4;
  return a4;
}
