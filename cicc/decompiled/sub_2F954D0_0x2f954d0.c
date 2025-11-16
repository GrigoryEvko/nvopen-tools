// Function: sub_2F954D0
// Address: 0x2f954d0
//
void __fastcall sub_2F954D0(_QWORD *a1)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  _DWORD *v8; // rax
  _DWORD *i; // rdx
  unsigned int *v10; // rax
  __int64 j; // r8
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // ecx
  unsigned __int64 *v17; // r15
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 *k; // r14
  __int64 v29; // r13
  __int64 v30; // rax
  unsigned int v31; // ebx
  __int64 v32; // rdx
  __int64 v33; // r9
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // r10
  __int64 v37; // r15
  __int64 v38; // rdi
  __int64 *v39; // rax
  __int64 v40; // rcx
  __int64 *v41; // rdx
  int v42; // edx
  __int64 v43; // rdi
  unsigned __int64 *v44; // rax
  unsigned __int64 *v45; // rdx
  unsigned __int64 v46; // r9
  unsigned __int64 v47; // rax
  unsigned int v48; // r9d
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // r14
  unsigned __int64 *v52; // r13
  __int64 v53; // [rsp+10h] [rbp-50h]
  unsigned int v54; // [rsp+18h] [rbp-48h]
  unsigned int v55; // [rsp+20h] [rbp-40h]
  __int64 v56; // [rsp+20h] [rbp-40h]
  __int64 v57; // [rsp+20h] [rbp-40h]
  __int64 v58; // [rsp+28h] [rbp-38h]

  sub_3157250(a1 + 1);
  v4 = *a1;
  v5 = *((unsigned int *)a1 + 14);
  v6 = *(unsigned int *)(*a1 + 40LL);
  if ( v5 != v6 )
  {
    if ( v5 >= v6 )
    {
      if ( v5 > *(unsigned int *)(v4 + 44) )
      {
        sub_C8D5F0(v4 + 32, (const void *)(v4 + 48), *((unsigned int *)a1 + 14), 8u, v2, v3);
        v6 = *(unsigned int *)(v4 + 40);
      }
      v7 = *(_QWORD *)(v4 + 32);
      v8 = (_DWORD *)(v7 + 8 * v6);
      for ( i = (_DWORD *)(v7 + 8 * v5); i != v8; v8 += 2 )
      {
        if ( v8 )
        {
          v8[1] = 0;
          *v8 = -1;
        }
      }
    }
    *(_DWORD *)(v4 + 40) = v5;
    v4 = *a1;
  }
  v10 = (unsigned int *)a1[11];
  for ( j = (__int64)&v10[3 * *((unsigned int *)a1 + 24)]; (unsigned int *)j != v10; v4 = *a1 )
  {
    v12 = a1[1];
    v13 = v10[1];
    v14 = *(unsigned int *)(v12 + 4LL * *v10);
    v15 = 8 * v14;
    if ( (_DWORD)v13 != -1 )
    {
      *(_DWORD *)(*(_QWORD *)(v4 + 32) + 8 * v14) = *(_DWORD *)(v12 + 4 * v13);
      v4 = *a1;
    }
    v16 = v10[2];
    v10 += 3;
    *(_DWORD *)(*(_QWORD *)(v4 + 32) + v15 + 4) = v16;
  }
  v17 = *(unsigned __int64 **)(v4 + 184);
  v18 = *(_QWORD *)(v4 + 176);
  v19 = *((unsigned int *)a1 + 14);
  v20 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v17 - v18) >> 4);
  if ( v19 > v20 )
  {
    sub_2F95200((unsigned __int64 *)(v4 + 176), v19 - v20, v18, 0xAAAAAAAAAAAAAAABLL, j, v3);
    v4 = *a1;
    v19 = *((unsigned int *)a1 + 14);
  }
  else if ( v19 < v20 )
  {
    v51 = v18 + 48 * v19;
    if ( v17 != (unsigned __int64 *)v51 )
    {
      v52 = (unsigned __int64 *)(v18 + 48 * v19);
      do
      {
        if ( (unsigned __int64 *)*v52 != v52 + 2 )
          _libc_free(*v52);
        v52 += 6;
      }
      while ( v17 != v52 );
      *(_QWORD *)(v4 + 184) = v51;
      v4 = *a1;
      v19 = *((unsigned int *)a1 + 14);
    }
  }
  v21 = *(_QWORD *)(v4 + 200);
  v22 = (__int64)(*(_QWORD *)(v4 + 208) - v21) >> 2;
  if ( v22 < v19 )
  {
    v19 -= v22;
    sub_C17A60(v4 + 200, v19);
    v4 = *a1;
  }
  else if ( v22 > v19 )
  {
    v50 = v21 + 4 * v19;
    if ( *(_QWORD *)(v4 + 208) != v50 )
    {
      *(_QWORD *)(v4 + 208) = v50;
      v4 = *a1;
    }
  }
  v23 = *(_QWORD *)(v4 + 8);
  v24 = (*(_QWORD *)(v4 + 16) - v23) >> 3;
  if ( (_DWORD)v24 )
  {
    v25 = (unsigned int)(v24 - 1);
    v26 = 0;
    v19 = 4 * v25;
    while ( 1 )
    {
      v21 = *(unsigned int *)(a1[1] + v26);
      *(_DWORD *)(v23 + 2 * v26 + 4) = v21;
      if ( v19 == v26 )
        break;
      v26 += 4;
      v23 = *(_QWORD *)(*a1 + 8LL);
    }
  }
  v27 = a1[9];
  for ( k = (__int64 *)a1[8]; (__int64 *)v27 != k; k += 2 )
  {
    v29 = *k;
    v30 = a1[1];
    v31 = *(_DWORD *)(v30 + 4LL * *(unsigned int *)(*k + 200));
    v32 = *(unsigned int *)(k[1] + 200);
    v33 = *(unsigned int *)(v30 + 4 * v32);
    if ( (_DWORD)v33 != v31 )
    {
      if ( (*(_BYTE *)(v29 + 254) & 1) == 0 )
      {
        v55 = *(_DWORD *)(v30 + 4 * v32);
        v58 = v27;
        sub_2F8F5D0(*k, (_QWORD *)v19, v32, v21, v27, v33);
        v33 = v55;
        v27 = v58;
      }
      v34 = *(unsigned int *)(v29 + 240);
      if ( (_DWORD)v34 )
      {
        v35 = *a1;
        v36 = v31;
        v37 = (unsigned int)v33;
        while ( 1 )
        {
          v38 = *(_QWORD *)(v35 + 176) + 48 * v36;
          v39 = *(__int64 **)v38;
          v40 = *(unsigned int *)(v38 + 8);
          v41 = (__int64 *)(*(_QWORD *)v38 + 8 * v40);
          if ( *(__int64 **)v38 != v41 )
            break;
LABEL_48:
          v49 = v37 | (v34 << 32);
          if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v38 + 12) )
          {
            v19 = v38 + 16;
            v53 = v36;
            v54 = v33;
            v56 = v27;
            sub_C8D5F0(v38, (const void *)(v38 + 16), v40 + 1, 8u, v27, v33);
            v49 = v37 | (v34 << 32);
            v36 = v53;
            v33 = v54;
            v27 = v56;
            v41 = (__int64 *)(*(_QWORD *)v38 + 8LL * *(unsigned int *)(v38 + 8));
          }
          *v41 = v49;
          ++*(_DWORD *)(v38 + 8);
          v35 = *a1;
          v36 = *(unsigned int *)(*(_QWORD *)(*a1 + 32LL) + 8 * v36);
          if ( (_DWORD)v36 == -1 )
            goto LABEL_35;
        }
        while ( (_DWORD)v33 != *(_DWORD *)v39 )
        {
          if ( v41 == ++v39 )
            goto LABEL_48;
        }
        v42 = v34;
        if ( *((_DWORD *)v39 + 1) >= (unsigned int)v34 )
          v42 = *((_DWORD *)v39 + 1);
        *((_DWORD *)v39 + 1) = v42;
        v35 = *a1;
LABEL_35:
        while ( 1 )
        {
          v43 = *(_QWORD *)(v35 + 176) + 48 * v37;
          v44 = *(unsigned __int64 **)v43;
          v21 = *(unsigned int *)(v43 + 8);
          v45 = (unsigned __int64 *)(*(_QWORD *)v43 + 8 * v21);
          if ( *(unsigned __int64 **)v43 != v45 )
            break;
LABEL_44:
          v46 = v21 + 1;
          v21 = *(unsigned int *)(v43 + 12);
          v19 = v31;
          v47 = v31 | (unsigned __int64)(v34 << 32);
          if ( v46 > v21 )
          {
            v19 = v43 + 16;
            v57 = v27;
            sub_C8D5F0(v43, (const void *)(v43 + 16), v46, 8u, v27, v46);
            v47 = v31 | (unsigned __int64)(v34 << 32);
            v27 = v57;
            v21 = *(unsigned int *)(v43 + 8);
            v45 = (unsigned __int64 *)(*(_QWORD *)v43 + 8 * v21);
          }
          *v45 = v47;
          ++*(_DWORD *)(v43 + 8);
          v35 = *a1;
          v48 = *(_DWORD *)(*(_QWORD *)(*a1 + 32LL) + 8 * v37);
          if ( v48 == -1 )
            goto LABEL_42;
          v37 = v48;
        }
        while ( v31 != *(_DWORD *)v44 )
        {
          if ( v45 == ++v44 )
            goto LABEL_44;
        }
        if ( *((_DWORD *)v44 + 1) >= (unsigned int)v34 )
          LODWORD(v34) = *((_DWORD *)v44 + 1);
        *((_DWORD *)v44 + 1) = v34;
      }
    }
LABEL_42:
    ;
  }
}
