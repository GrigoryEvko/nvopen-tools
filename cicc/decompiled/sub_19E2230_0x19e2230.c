// Function: sub_19E2230
// Address: 0x19e2230
//
__int64 __fastcall sub_19E2230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 i; // rcx
  int v9; // esi
  __int64 v10; // rax
  __int64 v11; // r8
  _QWORD *v12; // rdx
  __int64 v13; // r11
  __int64 v14; // r14
  int v15; // esi
  __int64 v16; // rbx
  unsigned int v17; // r10d
  __int64 *v18; // r8
  __int64 v19; // r12
  unsigned int v20; // r15d
  __int64 v21; // r11
  unsigned int v22; // r10d
  __int64 *v23; // r8
  __int64 v24; // r12
  _QWORD *v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // r13d
  int v28; // esi
  __int64 v29; // r11
  __int64 v30; // r12
  unsigned int v31; // r10d
  __int64 *v32; // rcx
  __int64 v33; // r15
  unsigned int v34; // r10d
  unsigned int v35; // r12d
  __int64 *v36; // rcx
  __int64 v37; // r14
  _QWORD *v38; // rax
  int v39; // esi
  __int64 v41; // rcx
  _QWORD *v42; // rcx
  int v43; // ecx
  int v44; // r15d
  int v45; // ecx
  int v46; // r8d
  int v47; // r14d
  int v48; // r8d
  int v49; // r15d
  int v50; // r14d
  __int64 v53; // [rsp+10h] [rbp-40h]

  v6 = a2;
  v7 = (a3 - 1) / 2;
  v53 = a3 & 1;
  if ( a2 >= v7 )
  {
    v10 = a2;
    v12 = (_QWORD *)(a1 + 16 * a2);
    if ( v53 )
      goto LABEL_22;
  }
  else
  {
    for ( i = a2; ; i = v10 )
    {
      v9 = *(_DWORD *)(a6 + 2384);
      v10 = 2 * (i + 1);
      v11 = 32 * (i + 1);
      v12 = (_QWORD *)(a1 + v11);
      if ( v9 )
      {
        v13 = v12[1];
        v14 = a1 + v11 - 16;
        v15 = v9 - 1;
        v16 = *(_QWORD *)(a6 + 2368);
        v17 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v13 == *v18 )
        {
LABEL_6:
          v20 = *((_DWORD *)v18 + 2);
          v21 = *(_QWORD *)(v14 + 8);
        }
        else
        {
          v48 = 1;
          while ( v19 != -8 )
          {
            v49 = v48 + 1;
            v17 = v15 & (v48 + v17);
            v18 = (__int64 *)(v16 + 16LL * v17);
            v19 = *v18;
            if ( v13 == *v18 )
              goto LABEL_6;
            v48 = v49;
          }
          v21 = *(_QWORD *)(v14 + 8);
          v20 = 0;
        }
        v22 = v15 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v23 = (__int64 *)(v16 + 16LL * v22);
        v24 = *v23;
        if ( *v23 == v21 )
        {
LABEL_8:
          if ( *((_DWORD *)v23 + 2) > v20 )
          {
            --v10;
            v12 = (_QWORD *)(a1 + 16 * v10);
          }
        }
        else
        {
          v46 = 1;
          while ( v24 != -8 )
          {
            v47 = v46 + 1;
            v22 = v15 & (v46 + v22);
            v23 = (__int64 *)(v16 + 16LL * v22);
            v24 = *v23;
            if ( *v23 == v21 )
              goto LABEL_8;
            v46 = v47;
          }
        }
      }
      v25 = (_QWORD *)(a1 + 16 * i);
      *v25 = *v12;
      v25[1] = v12[1];
      if ( v10 >= v7 )
        break;
    }
    v6 = a2;
    if ( v53 )
      goto LABEL_12;
  }
  if ( (a3 - 2) / 2 == v10 )
  {
    v41 = v10 + 1;
    v10 = 2 * (v10 + 1) - 1;
    v42 = (_QWORD *)(a1 + 32 * v41 - 16);
    *v12 = *v42;
    v12[1] = v42[1];
    v12 = (_QWORD *)(a1 + 16 * v10);
  }
LABEL_12:
  v26 = (v10 - 1) / 2;
  if ( v10 > v6 )
  {
    v27 = ((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4);
    while ( 1 )
    {
      v39 = *(_DWORD *)(a6 + 2384);
      if ( !v39 )
        goto LABEL_21;
      v28 = v39 - 1;
      v29 = *(_QWORD *)(a6 + 2368);
      v12 = (_QWORD *)(a1 + 16 * v26);
      v30 = v12[1];
      v31 = v28 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v32 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v32;
      if ( v30 == *v32 )
      {
LABEL_15:
        v34 = *((_DWORD *)v32 + 2);
      }
      else
      {
        v45 = 1;
        while ( v33 != -8 )
        {
          v50 = v45 + 1;
          v31 = v28 & (v45 + v31);
          v32 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v32;
          if ( v30 == *v32 )
            goto LABEL_15;
          v45 = v50;
        }
        v34 = 0;
      }
      v35 = v27 & v28;
      v36 = (__int64 *)(v29 + 16LL * (v27 & v28));
      v37 = *v36;
      if ( *v36 != a5 )
      {
        v43 = 1;
        while ( v37 != -8 )
        {
          v44 = v43 + 1;
          v35 = v28 & (v43 + v35);
          v36 = (__int64 *)(v29 + 16LL * v35);
          v37 = *v36;
          if ( *v36 == a5 )
            goto LABEL_17;
          v43 = v44;
        }
LABEL_21:
        v12 = (_QWORD *)(a1 + 16 * v10);
        goto LABEL_22;
      }
LABEL_17:
      v38 = (_QWORD *)(a1 + 16 * v10);
      if ( *((_DWORD *)v36 + 2) <= v34 )
        break;
      *v38 = *v12;
      v38[1] = v12[1];
      v10 = v26;
      if ( v6 >= v26 )
        goto LABEL_22;
      v26 = (v26 - 1) / 2;
    }
    v12 = v38;
  }
LABEL_22:
  *v12 = a4;
  v12[1] = a5;
  return a5;
}
