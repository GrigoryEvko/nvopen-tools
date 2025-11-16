// Function: sub_2F70DF0
// Address: 0x2f70df0
//
__int64 *__fastcall sub_2F70DF0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *result; // rax
  __int64 v4; // r15
  __int64 *v5; // r14
  char *v7; // r8
  __int64 *v8; // rbx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  unsigned int v11; // edx
  unsigned int v12; // esi
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  char *v17; // r13
  __int64 *v18; // rdi
  int v19; // esi
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // ecx
  __int64 *i; // rdx
  unsigned int v26; // eax
  unsigned int v27; // esi
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  __int64 *v38; // rsi
  __int64 *v39; // r14
  __int64 v40; // rcx
  unsigned __int64 v41; // r8
  __int64 v42; // rbx

  result = (__int64 *)((char *)a2 - (char *)a1);
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = a3;
  v5 = a2;
  if ( !a3 )
    goto LABEL_37;
  v7 = (char *)a2;
  v8 = a1 + 2;
  while ( 2 )
  {
    --v4;
    v9 = *(_DWORD *)((a1[2] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a1[2] >> 1) & 3;
    v10 = &a1[2 * ((__int64)(((v7 - (char *)a1) >> 4) + ((unsigned __int64)(v7 - (char *)a1) >> 63)) >> 1)];
    v11 = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v10 >> 1) & 3;
    if ( v9 < v11 || v9 <= v11 && a1[3] < (unsigned __int64)v10[1] )
    {
      v27 = *(_DWORD *)((*((_QWORD *)v7 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*((__int64 *)v7 - 2) >> 1) & 3;
      if ( v11 < v27 || v11 <= v27 && (unsigned __int64)v10[1] < *((_QWORD *)v7 - 1) )
        goto LABEL_36;
      if ( v9 >= v27 )
      {
        v28 = a1[3];
        if ( v9 > v27 || *((_QWORD *)v7 - 1) <= v28 )
        {
          v32 = a1[2];
          a1[2] = *a1;
          v33 = a1[1];
          *a1 = v32;
          a1[1] = v28;
          a1[3] = v33;
          goto LABEL_10;
        }
      }
      goto LABEL_26;
    }
    v12 = *(_DWORD *)((*((_QWORD *)v7 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*((__int64 *)v7 - 2) >> 1) & 3;
    if ( v9 >= v12 )
    {
      if ( v9 <= v12 )
      {
        v13 = a1[3];
        if ( v13 < *((_QWORD *)v7 - 1) )
          goto LABEL_9;
      }
      if ( v11 >= v12 && (v11 > v12 || (unsigned __int64)v10[1] >= *((_QWORD *)v7 - 1)) )
      {
LABEL_36:
        v34 = *a1;
        *a1 = *v10;
        v35 = v10[1];
        *v10 = v34;
        v36 = a1[1];
        a1[1] = v35;
        v10[1] = v36;
        goto LABEL_10;
      }
LABEL_26:
      v29 = *a1;
      *a1 = *((_QWORD *)v7 - 2);
      v30 = *((_QWORD *)v7 - 1);
      *((_QWORD *)v7 - 2) = v29;
      v31 = a1[1];
      a1[1] = v30;
      *((_QWORD *)v7 - 1) = v31;
      goto LABEL_10;
    }
    v13 = a1[3];
LABEL_9:
    v14 = a1[2];
    a1[2] = *a1;
    v15 = a1[1];
    *a1 = v14;
    a1[1] = v13;
    a1[3] = v15;
LABEL_10:
    v16 = *a1;
    v17 = (char *)v8;
    v18 = (__int64 *)v7;
    v19 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    while ( 1 )
    {
      v5 = (__int64 *)v17;
      v23 = *(_DWORD *)((*(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)v17 >> 1) & 3;
      v24 = v19 | (v16 >> 1) & 3;
      if ( v23 >= v24 && (v23 > v24 || *((_QWORD *)v17 + 1) >= (unsigned __int64)a1[1]) )
        break;
LABEL_14:
      v17 += 16;
    }
    for ( i = v18 - 2; ; i -= 2 )
    {
      while ( 1 )
      {
        v18 = i;
        v26 = *(_DWORD *)((*i & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*i >> 1) & 3;
        if ( v24 >= v26 )
          break;
        i -= 2;
      }
      if ( v24 > v26 || a1[1] >= (unsigned __int64)i[1] )
        break;
    }
    if ( v17 < (char *)i )
    {
      v20 = *(_QWORD *)v17;
      *(_QWORD *)v17 = *i;
      v21 = i[1];
      *i = v20;
      v22 = *((_QWORD *)v17 + 1);
      *((_QWORD *)v17 + 1) = v21;
      i[1] = v22;
      v16 = *a1;
      v19 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      goto LABEL_14;
    }
    sub_2F70DF0(v17, v7, v4);
    result = (__int64 *)(v17 - (char *)a1);
    if ( v17 - (char *)a1 > 256 )
    {
      if ( v4 )
      {
        v7 = v17;
        continue;
      }
LABEL_37:
      v37 = (unsigned __int64)v5;
      v38 = v5;
      v39 = v5 - 2;
      sub_2F70D00(a1, v38, v37);
      do
      {
        v40 = *v39;
        v41 = v39[1];
        v42 = (char *)v39 - (char *)a1;
        v39 -= 2;
        v39[2] = *a1;
        v39[3] = a1[1];
        result = sub_2F621E0((__int64)a1, 0, v42 >> 4, v40, v41);
      }
      while ( v42 > 16 );
    }
    return result;
  }
}
