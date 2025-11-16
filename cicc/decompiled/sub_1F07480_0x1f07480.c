// Function: sub_1F07480
// Address: 0x1f07480
//
void __fastcall sub_1F07480(_QWORD *a1)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned int *v7; // rax
  __int64 j; // r8
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdi
  unsigned int v13; // ecx
  unsigned __int64 *v14; // r15
  __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 *k; // r14
  __int64 v26; // r13
  __int64 v27; // rax
  unsigned int v28; // ebx
  __int64 v29; // rdx
  unsigned int v30; // r9d
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // r15
  __int64 v35; // rdi
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 *v38; // rdx
  int v39; // edx
  __int64 v40; // rdi
  unsigned __int64 *v41; // rax
  unsigned __int64 *v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  _DWORD *v46; // rax
  _DWORD *i; // rdx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // r14
  unsigned __int64 *v50; // r13
  __int64 v51; // [rsp+10h] [rbp-50h]
  unsigned int v52; // [rsp+18h] [rbp-48h]
  unsigned int v53; // [rsp+20h] [rbp-40h]
  __int64 v54; // [rsp+20h] [rbp-40h]
  __int64 v55; // [rsp+20h] [rbp-40h]
  __int64 v56; // [rsp+28h] [rbp-38h]

  sub_3945BD0(a1 + 1);
  v4 = *a1;
  v5 = *((unsigned int *)a1 + 14);
  v6 = *(unsigned int *)(*a1 + 40LL);
  if ( v5 >= v6 )
  {
    if ( v5 <= v6 )
      goto LABEL_3;
    if ( v5 > *(unsigned int *)(v4 + 44) )
    {
      sub_16CD150(v4 + 32, (const void *)(v4 + 48), *((unsigned int *)a1 + 14), 8, v2, v3);
      v6 = *(unsigned int *)(v4 + 40);
    }
    v45 = *(_QWORD *)(v4 + 32);
    v46 = (_DWORD *)(v45 + 8 * v6);
    for ( i = (_DWORD *)(v45 + 8 * v5); i != v46; v46 += 2 )
    {
      if ( v46 )
      {
        v46[1] = 0;
        *v46 = -1;
      }
    }
  }
  *(_DWORD *)(v4 + 40) = v5;
  v4 = *a1;
LABEL_3:
  v7 = (unsigned int *)a1[11];
  for ( j = (__int64)&v7[3 * *((unsigned int *)a1 + 24)]; (unsigned int *)j != v7; v4 = *a1 )
  {
    v9 = a1[1];
    v10 = v7[1];
    v11 = *(unsigned int *)(v9 + 4LL * *v7);
    v12 = 8 * v11;
    if ( (_DWORD)v10 != -1 )
    {
      *(_DWORD *)(*(_QWORD *)(v4 + 32) + 8 * v11) = *(_DWORD *)(v9 + 4 * v10);
      v4 = *a1;
    }
    v13 = v7[2];
    v7 += 3;
    *(_DWORD *)(*(_QWORD *)(v4 + 32) + v12 + 4) = v13;
  }
  v14 = *(unsigned __int64 **)(v4 + 184);
  v15 = *(_QWORD *)(v4 + 176);
  v16 = *((unsigned int *)a1 + 14);
  v17 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v14 - v15) >> 4);
  if ( v16 > v17 )
  {
    sub_1F071B0((__int64 *)(v4 + 176), v16 - v17, v15, 0xAAAAAAAAAAAAAAABLL, j, v3);
    v4 = *a1;
    v16 = *((unsigned int *)a1 + 14);
  }
  else if ( v16 < v17 )
  {
    v49 = v15 + 48 * v16;
    if ( v14 != (unsigned __int64 *)v49 )
    {
      v50 = (unsigned __int64 *)(v15 + 48 * v16);
      do
      {
        if ( (unsigned __int64 *)*v50 != v50 + 2 )
          _libc_free(*v50);
        v50 += 6;
      }
      while ( v14 != v50 );
      *(_QWORD *)(v4 + 184) = v49;
      v4 = *a1;
      v16 = *((unsigned int *)a1 + 14);
    }
  }
  v18 = *(_QWORD *)(v4 + 200);
  v19 = (*(_QWORD *)(v4 + 208) - v18) >> 2;
  if ( v19 < v16 )
  {
    v16 -= v19;
    sub_C17A60(v4 + 200, v16);
    v4 = *a1;
  }
  else if ( v19 > v16 )
  {
    v48 = v18 + 4 * v16;
    if ( *(_QWORD *)(v4 + 208) != v48 )
    {
      *(_QWORD *)(v4 + 208) = v48;
      v4 = *a1;
    }
  }
  v20 = *(_QWORD *)(v4 + 8);
  v21 = (*(_QWORD *)(v4 + 16) - v20) >> 3;
  if ( (_DWORD)v21 )
  {
    v22 = (unsigned int)(v21 - 1);
    v23 = 0;
    v16 = 4 * v22;
    while ( 1 )
    {
      v18 = *(unsigned int *)(a1[1] + v23);
      *(_DWORD *)(v20 + 2 * v23 + 4) = v18;
      if ( v16 == v23 )
        break;
      v23 += 4;
      v20 = *(_QWORD *)(*a1 + 8LL);
    }
  }
  v24 = a1[9];
  for ( k = (__int64 *)a1[8]; (__int64 *)v24 != k; k += 2 )
  {
    v26 = *k;
    v27 = a1[1];
    v28 = *(_DWORD *)(v27 + 4LL * *(unsigned int *)(*k + 192));
    v29 = *(unsigned int *)(k[1] + 192);
    v30 = *(_DWORD *)(v27 + 4 * v29);
    if ( v30 != v28 )
    {
      if ( (*(_BYTE *)(v26 + 236) & 1) == 0 )
      {
        v53 = *(_DWORD *)(v27 + 4 * v29);
        v56 = v24;
        sub_1F01DD0(*k, (_QWORD *)v16, v29, v18, v24, v30);
        v30 = v53;
        v24 = v56;
      }
      v31 = *(unsigned int *)(v26 + 240);
      if ( (_DWORD)v31 )
      {
        v32 = *a1;
        v33 = v28;
        v34 = v30;
        while ( 1 )
        {
          v35 = *(_QWORD *)(v32 + 176) + 48 * v33;
          v36 = *(__int64 **)v35;
          v37 = *(unsigned int *)(v35 + 8);
          v38 = (__int64 *)(*(_QWORD *)v35 + 8 * v37);
          if ( *(__int64 **)v35 != v38 )
            break;
LABEL_41:
          v44 = v34 | (v31 << 32);
          if ( (unsigned int)v37 >= *(_DWORD *)(v35 + 12) )
          {
            v16 = v35 + 16;
            v51 = v33;
            v52 = v30;
            v54 = v24;
            sub_16CD150(v35, (const void *)(v35 + 16), 0, 8, v24, v30);
            v44 = v34 | (v31 << 32);
            v33 = v51;
            v30 = v52;
            v24 = v54;
            v38 = (__int64 *)(*(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8));
          }
          *v38 = v44;
          ++*(_DWORD *)(v35 + 8);
          v32 = *a1;
          v33 = *(unsigned int *)(*(_QWORD *)(*a1 + 32LL) + 8 * v33);
          if ( (_DWORD)v33 == -1 )
            goto LABEL_28;
        }
        while ( v30 != *(_DWORD *)v36 )
        {
          if ( v38 == ++v36 )
            goto LABEL_41;
        }
        v39 = v31;
        if ( *((_DWORD *)v36 + 1) >= (unsigned int)v31 )
          v39 = *((_DWORD *)v36 + 1);
        *((_DWORD *)v36 + 1) = v39;
        v32 = *a1;
LABEL_28:
        while ( 1 )
        {
          v40 = *(_QWORD *)(v32 + 176) + 48 * v34;
          v41 = *(unsigned __int64 **)v40;
          v18 = *(unsigned int *)(v40 + 8);
          v42 = (unsigned __int64 *)(*(_QWORD *)v40 + 8 * v18);
          if ( *(unsigned __int64 **)v40 != v42 )
            break;
LABEL_37:
          v16 = v28;
          v43 = v28 | (unsigned __int64)(v31 << 32);
          if ( (unsigned int)v18 >= *(_DWORD *)(v40 + 12) )
          {
            v16 = v40 + 16;
            v55 = v24;
            sub_16CD150(v40, (const void *)(v40 + 16), 0, 8, v24, v30);
            v43 = v28 | (unsigned __int64)(v31 << 32);
            v24 = v55;
            v18 = *(unsigned int *)(v40 + 8);
            v42 = (unsigned __int64 *)(*(_QWORD *)v40 + 8 * v18);
          }
          *v42 = v43;
          ++*(_DWORD *)(v40 + 8);
          v32 = *a1;
          v30 = *(_DWORD *)(*(_QWORD *)(*a1 + 32LL) + 8 * v34);
          if ( v30 == -1 )
            goto LABEL_35;
          v34 = v30;
        }
        while ( v28 != *(_DWORD *)v41 )
        {
          if ( v42 == ++v41 )
            goto LABEL_37;
        }
        if ( *((_DWORD *)v41 + 1) >= (unsigned int)v31 )
          LODWORD(v31) = *((_DWORD *)v41 + 1);
        *((_DWORD *)v41 + 1) = v31;
      }
    }
LABEL_35:
    ;
  }
}
