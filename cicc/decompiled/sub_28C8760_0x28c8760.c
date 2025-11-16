// Function: sub_28C8760
// Address: 0x28c8760
//
__int64 __fastcall sub_28C8760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 i; // rcx
  int v9; // r8d
  __int64 v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // r11
  __int64 v14; // rbx
  int v15; // r8d
  unsigned int v16; // r10d
  __int64 *v17; // rsi
  __int64 v18; // r14
  unsigned int v19; // r14d
  unsigned int v20; // r10d
  __int64 *v21; // rsi
  __int64 v22; // rbx
  _QWORD *v23; // rcx
  __int64 v24; // r10
  unsigned int v25; // r12d
  int v26; // esi
  unsigned int v27; // r8d
  __int64 *v28; // rcx
  __int64 v29; // r13
  unsigned int v30; // r8d
  unsigned int v31; // r13d
  __int64 *v32; // rcx
  __int64 v33; // r11
  _QWORD *v34; // rax
  int v35; // esi
  __int64 v36; // rbx
  __int64 v37; // r11
  int v39; // ecx
  int v40; // ecx
  int v41; // esi
  int v42; // esi
  __int64 v43; // rcx
  _QWORD *v44; // rcx
  int v45; // [rsp+0h] [rbp-4Ch]
  int v46; // [rsp+0h] [rbp-4Ch]
  __int64 v48; // [rsp+Ch] [rbp-40h]
  int v49; // [rsp+Ch] [rbp-40h]
  int v50; // [rsp+Ch] [rbp-40h]

  v7 = (a3 - 1) / 2;
  v48 = a3 & 1;
  if ( a2 >= v7 )
  {
    v11 = a2;
    v12 = (_QWORD *)(a1 + 16 * a2);
    if ( v48 )
      goto LABEL_22;
  }
  else
  {
    for ( i = a2; ; i = v11 )
    {
      v9 = *(_DWORD *)(a6 + 2376);
      v10 = *(_QWORD *)(a6 + 2360);
      v11 = 2 * (i + 1);
      v12 = (_QWORD *)(a1 + 32 * (i + 1));
      v13 = *(v12 - 1);
      v14 = v12[1];
      if ( v9 )
      {
        v15 = v9 - 1;
        v16 = v15 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v17 = (__int64 *)(v10 + 16LL * v16);
        v18 = *v17;
        if ( v14 == *v17 )
        {
LABEL_6:
          v19 = *((_DWORD *)v17 + 2);
        }
        else
        {
          v42 = 1;
          while ( v18 != -4096 )
          {
            v16 = v15 & (v42 + v16);
            v46 = v42 + 1;
            v17 = (__int64 *)(v10 + 16LL * v16);
            v18 = *v17;
            if ( v14 == *v17 )
              goto LABEL_6;
            v42 = v46;
          }
          v19 = 0;
        }
        v20 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v21 = (__int64 *)(v10 + 16LL * v20);
        v22 = *v21;
        if ( v13 == *v21 )
        {
LABEL_8:
          if ( *((_DWORD *)v21 + 2) > v19 )
          {
            --v11;
            v12 = (_QWORD *)(a1 + 16 * v11);
          }
        }
        else
        {
          v41 = 1;
          while ( v22 != -4096 )
          {
            v20 = v15 & (v41 + v20);
            v45 = v41 + 1;
            v21 = (__int64 *)(v10 + 16LL * v20);
            v22 = *v21;
            if ( v13 == *v21 )
              goto LABEL_8;
            v41 = v45;
          }
        }
      }
      v23 = (_QWORD *)(a1 + 16 * i);
      *v23 = *v12;
      v23[1] = v12[1];
      if ( v11 >= v7 )
        break;
    }
    if ( v48 )
      goto LABEL_12;
  }
  if ( (a3 - 2) / 2 == v11 )
  {
    v43 = v11 + 1;
    v11 = 2 * (v11 + 1) - 1;
    v44 = (_QWORD *)(a1 + 32 * v43 - 16);
    *v12 = *v44;
    v12[1] = v44[1];
    v12 = (_QWORD *)(a1 + 16 * v11);
  }
LABEL_12:
  v24 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    v25 = ((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4);
    while ( 1 )
    {
      v35 = *(_DWORD *)(a6 + 2376);
      v36 = *(_QWORD *)(a6 + 2360);
      v12 = (_QWORD *)(a1 + 16 * v24);
      v37 = v12[1];
      if ( !v35 )
        goto LABEL_21;
      v26 = v35 - 1;
      v27 = v26 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v28 = (__int64 *)(v36 + 16LL * v27);
      v29 = *v28;
      if ( v37 == *v28 )
      {
LABEL_15:
        v30 = *((_DWORD *)v28 + 2);
      }
      else
      {
        v40 = 1;
        while ( v29 != -4096 )
        {
          v27 = v26 & (v40 + v27);
          v50 = v40 + 1;
          v28 = (__int64 *)(v36 + 16LL * v27);
          v29 = *v28;
          if ( v37 == *v28 )
            goto LABEL_15;
          v40 = v50;
        }
        v30 = 0;
      }
      v31 = v25 & v26;
      v32 = (__int64 *)(v36 + 16LL * (v25 & v26));
      v33 = *v32;
      if ( *v32 != a5 )
      {
        v39 = 1;
        while ( v33 != -4096 )
        {
          v31 = v26 & (v39 + v31);
          v49 = v39 + 1;
          v32 = (__int64 *)(v36 + 16LL * v31);
          v33 = *v32;
          if ( *v32 == a5 )
            goto LABEL_17;
          v39 = v49;
        }
LABEL_21:
        v12 = (_QWORD *)(a1 + 16 * v11);
        goto LABEL_22;
      }
LABEL_17:
      v34 = (_QWORD *)(a1 + 16 * v11);
      if ( v30 >= *((_DWORD *)v32 + 2) )
        break;
      *v34 = *v12;
      v34[1] = v12[1];
      v11 = v24;
      if ( a2 >= v24 )
        goto LABEL_22;
      v24 = (v24 - 1) / 2;
    }
    v12 = v34;
  }
LABEL_22:
  *v12 = a4;
  v12[1] = a5;
  return a5;
}
