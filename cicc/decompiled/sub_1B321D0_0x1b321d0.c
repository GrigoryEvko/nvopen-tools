// Function: sub_1B321D0
// Address: 0x1b321d0
//
__int64 __fastcall sub_1B321D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r11
  __int64 v7; // r15
  int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // r9
  int v13; // ecx
  __int64 v14; // r12
  unsigned int v15; // ebx
  __int64 *v16; // r10
  __int64 v17; // r13
  unsigned int v18; // r14d
  __int64 v19; // rbx
  unsigned int v20; // r10d
  __int64 *v21; // rsi
  __int64 v22; // r13
  __int64 v23; // r9
  unsigned int v24; // r14d
  int v25; // esi
  __int64 v26; // r11
  __int64 v27; // r10
  unsigned int v28; // ebx
  __int64 *v29; // rcx
  __int64 v30; // r13
  unsigned int v31; // ebx
  unsigned int v32; // r13d
  __int64 *v33; // rcx
  __int64 v34; // r12
  __int64 *v35; // rax
  int v36; // esi
  __int64 v38; // rax
  __int64 v39; // rcx
  int v40; // ecx
  int v41; // ecx
  int v42; // esi
  int v43; // r10d
  int v44; // r14d
  int v45; // r12d
  int v46; // [rsp+0h] [rbp-4Ch]
  __int64 v49; // [rsp+14h] [rbp-38h]
  int v50; // [rsp+14h] [rbp-38h]

  v5 = a2;
  v49 = a3 & 1;
  if ( a2 >= (a3 - 1) / 2 )
  {
    v11 = (__int64 *)(a1 + 8 * a2);
    v9 = a2;
    if ( v49 )
      goto LABEL_22;
  }
  else
  {
    v6 = a2;
    v7 = (a3 - 1) / 2;
    while ( 1 )
    {
      v8 = *(_DWORD *)(a5 + 920);
      v9 = 2 * (v6 + 1);
      v10 = 16 * (v6 + 1);
      v11 = (__int64 *)(a1 + v10);
      v12 = *(_QWORD *)(a1 + v10);
      if ( v8 )
      {
        v13 = v8 - 1;
        v14 = *(_QWORD *)(a5 + 904);
        v15 = v13 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
        {
LABEL_6:
          v18 = *((_DWORD *)v16 + 2);
        }
        else
        {
          v43 = 1;
          while ( v17 != -8 )
          {
            v44 = v43 + 1;
            v15 = v13 & (v43 + v15);
            v16 = (__int64 *)(v14 + 16LL * v15);
            v17 = *v16;
            if ( v12 == *v16 )
              goto LABEL_6;
            v43 = v44;
          }
          v18 = 0;
        }
        v19 = *(_QWORD *)(a1 + v10 - 8);
        v20 = v13 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v21 = (__int64 *)(v14 + 16LL * v20);
        v22 = *v21;
        if ( v19 == *v21 )
        {
LABEL_8:
          if ( *((_DWORD *)v21 + 2) > v18 )
          {
            --v9;
            v11 = (__int64 *)(a1 + 8 * v9);
            v12 = *v11;
          }
        }
        else
        {
          v42 = 1;
          while ( v22 != -8 )
          {
            v20 = v13 & (v42 + v20);
            v46 = v42 + 1;
            v21 = (__int64 *)(v14 + 16LL * v20);
            v22 = *v21;
            if ( v19 == *v21 )
              goto LABEL_8;
            v42 = v46;
          }
        }
      }
      *(_QWORD *)(a1 + 8 * v6) = v12;
      if ( v9 >= v7 )
        break;
      v6 = v9;
    }
    v5 = a2;
    if ( v49 )
      goto LABEL_12;
  }
  if ( (a3 - 2) / 2 == v9 )
  {
    v38 = 2 * v9 + 2;
    v39 = *(_QWORD *)(a1 + 8 * v38 - 8);
    v9 = v38 - 1;
    *v11 = v39;
    v11 = (__int64 *)(a1 + 8 * v9);
  }
LABEL_12:
  v23 = (v9 - 1) / 2;
  if ( v9 > v5 )
  {
    v24 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
    while ( 1 )
    {
      v36 = *(_DWORD *)(a5 + 920);
      if ( !v36 )
        goto LABEL_21;
      v11 = (__int64 *)(a1 + 8 * v23);
      v25 = v36 - 1;
      v26 = *(_QWORD *)(a5 + 904);
      v27 = *v11;
      v28 = v25 & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
      v29 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v29;
      if ( *v29 == *v11 )
      {
LABEL_15:
        v31 = *((_DWORD *)v29 + 2);
      }
      else
      {
        v41 = 1;
        while ( v30 != -8 )
        {
          v45 = v41 + 1;
          v28 = v25 & (v41 + v28);
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( v27 == *v29 )
            goto LABEL_15;
          v41 = v45;
        }
        v31 = 0;
      }
      v32 = v24 & v25;
      v33 = (__int64 *)(v26 + 16LL * (v24 & v25));
      v34 = *v33;
      if ( a4 != *v33 )
      {
        v40 = 1;
        while ( v34 != -8 )
        {
          v32 = v25 & (v40 + v32);
          v50 = v40 + 1;
          v33 = (__int64 *)(v26 + 16LL * v32);
          v34 = *v33;
          if ( a4 == *v33 )
            goto LABEL_17;
          v40 = v50;
        }
LABEL_21:
        v11 = (__int64 *)(a1 + 8 * v9);
        goto LABEL_22;
      }
LABEL_17:
      v35 = (__int64 *)(a1 + 8 * v9);
      if ( *((_DWORD *)v33 + 2) <= v31 )
        break;
      *v35 = v27;
      v9 = v23;
      if ( v5 >= v23 )
        goto LABEL_22;
      v23 = (v23 - 1) / 2;
    }
    v11 = v35;
  }
LABEL_22:
  *v11 = a4;
  return a4;
}
