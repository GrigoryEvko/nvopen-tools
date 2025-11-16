// Function: sub_E424D0
// Address: 0xe424d0
//
__int64 __fastcall sub_E424D0(__int64 a1, _QWORD *a2, size_t a3)
{
  __int64 v5; // r13
  _QWORD *v6; // rbx
  _QWORD *v7; // r13
  _QWORD *v8; // rdi
  __int64 *v9; // r15
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdi
  unsigned int v14; // r13d
  __int64 v15; // rdi
  int v16; // r13d
  __int64 v17; // rcx
  __int64 v18; // rbx
  int v19; // r13d
  int v20; // eax
  int v21; // r10d
  __int64 v22; // r9
  unsigned int i; // r8d
  const void *v24; // r15
  unsigned int v25; // r8d
  __int64 v26; // rbx
  unsigned int v27; // r13d
  int v28; // eax
  int v29; // r10d
  unsigned int j; // r9d
  __int64 v31; // r8
  const void *v32; // r15
  int v33; // eax
  bool v34; // al
  unsigned int v36; // r9d
  int v37; // eax
  int v38; // eax
  int v39; // r13d
  __int64 v40; // rbx
  int v41; // r13d
  int v42; // eax
  int v43; // r10d
  unsigned int v44; // r8d
  const void *v45; // r15
  unsigned int v46; // r8d
  int v47; // eax
  bool v48; // al
  int v49; // eax
  bool v50; // al
  __int64 v51; // [rsp+8h] [rbp-58h]
  int v52; // [rsp+8h] [rbp-58h]
  int v53; // [rsp+8h] [rbp-58h]
  __int64 v54; // [rsp+10h] [rbp-50h]
  __int64 v55; // [rsp+10h] [rbp-50h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  int v57; // [rsp+1Ch] [rbp-44h]
  unsigned int v58; // [rsp+1Ch] [rbp-44h]
  unsigned int v59; // [rsp+1Ch] [rbp-44h]
  unsigned int v60; // [rsp+20h] [rbp-40h]
  __int64 v61; // [rsp+20h] [rbp-40h]
  __int64 v62; // [rsp+20h] [rbp-40h]

  v5 = *(unsigned int *)(a1 + 1304);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 1296);
    v7 = &v6[4 * v5];
    do
    {
      v8 = v6;
      if ( (v6[3] & 2) == 0 )
        v8 = (_QWORD *)*v6;
      (*(void (__fastcall **)(_QWORD *))(v6[3] & 0xFFFFFFFFFFFFFFF8LL))(v8);
      v6 += 4;
    }
    while ( v7 != v6 );
    v9 = *(__int64 **)(a1 + 1296);
    v10 = &v9[4 * *(unsigned int *)(a1 + 1304)];
    while ( v9 != v10 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 1);
        v10 -= 4;
        if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          break;
        v12 = (v11 >> 1) & 1;
        if ( (v11 & 4) != 0 )
        {
          v13 = (__int64)v10;
          if ( !(_BYTE)v12 )
            v13 = *v10;
          (*(void (__fastcall **)(__int64))((v11 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v13);
        }
        if ( (_BYTE)v12 )
          break;
        sub_C7D6A0(*v10, v10[1], v10[2]);
        if ( v9 == v10 )
          goto LABEL_15;
      }
    }
LABEL_15:
    *(_DWORD *)(a1 + 1304) = 0;
  }
  v14 = *(_DWORD *)(a1 + 1464);
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 1440);
    v15 = a1 + 1440;
LABEL_18:
    sub_E41C70(v15, 2 * v14);
    v16 = *(_DWORD *)(a1 + 1464);
    v17 = 0;
    if ( !v16 )
      goto LABEL_41;
    v18 = *(_QWORD *)(a1 + 1448);
    v19 = v16 - 1;
    v20 = sub_C94890(a2, a3);
    v21 = 1;
    v22 = 0;
    for ( i = v19 & v20; ; i = v19 & v25 )
    {
      v17 = v18 + 48LL * i;
      v24 = *(const void **)v17;
      if ( *(_QWORD *)v17 == -1 )
        goto LABEL_38;
      if ( v24 == (const void *)-2LL )
      {
        v48 = (_QWORD *)((char *)a2 + 2) == 0;
      }
      else
      {
        if ( *(_QWORD *)(v17 + 8) != a3 )
          goto LABEL_23;
        v52 = v21;
        v55 = v22;
        v58 = i;
        if ( !a3 )
          goto LABEL_41;
        v61 = v18 + 48LL * i;
        v47 = memcmp(a2, v24, a3);
        v17 = v61;
        i = v58;
        v22 = v55;
        v21 = v52;
        v48 = v47 == 0;
      }
      if ( v48 )
        goto LABEL_41;
      if ( !v22 && v24 == (const void *)-2LL )
        v22 = v17;
LABEL_23:
      v25 = v21 + i;
      ++v21;
    }
  }
  v26 = *(_QWORD *)(a1 + 1448);
  v27 = v14 - 1;
  v28 = sub_C94890(a2, a3);
  v29 = 1;
  v17 = 0;
  for ( j = v27 & v28; ; j = v27 & v36 )
  {
    v31 = v26 + 48LL * j;
    v32 = *(const void **)v31;
    if ( *(_QWORD *)v31 == -1 )
    {
      v34 = (_QWORD *)((char *)a2 + 1) == 0;
    }
    else if ( v32 == (const void *)-2LL )
    {
      v34 = (_QWORD *)((char *)a2 + 2) == 0;
    }
    else
    {
      if ( *(_QWORD *)(v31 + 8) != a3 )
        goto LABEL_34;
      v54 = v17;
      v57 = v29;
      v60 = j;
      if ( !a3 )
        return *(_QWORD *)(v31 + 16);
      v51 = v26 + 48LL * j;
      v33 = memcmp(a2, v32, a3);
      v31 = v51;
      j = v60;
      v29 = v57;
      v17 = v54;
      v34 = v33 == 0;
    }
    if ( v34 )
      return *(_QWORD *)(v31 + 16);
    if ( v32 == (const void *)-1LL )
      break;
LABEL_34:
    if ( v32 == (const void *)-2LL && !v17 )
      v17 = v31;
    v36 = v29 + j;
    ++v29;
  }
  v38 = *(_DWORD *)(a1 + 1456);
  v14 = *(_DWORD *)(a1 + 1464);
  v15 = a1 + 1440;
  if ( !v17 )
    v17 = v31;
  ++*(_QWORD *)(a1 + 1440);
  v37 = v38 + 1;
  if ( 4 * v37 >= 3 * v14 )
    goto LABEL_18;
  if ( v14 - (v37 + *(_DWORD *)(a1 + 1460)) > v14 >> 3 )
    goto LABEL_42;
  sub_E41C70(v15, v14);
  v39 = *(_DWORD *)(a1 + 1464);
  v17 = 0;
  if ( !v39 )
    goto LABEL_41;
  v40 = *(_QWORD *)(a1 + 1448);
  v41 = v39 - 1;
  v42 = sub_C94890(a2, a3);
  v43 = 1;
  v22 = 0;
  v44 = v41 & v42;
  while ( 2 )
  {
    v17 = v40 + 48LL * v44;
    v45 = *(const void **)v17;
    if ( *(_QWORD *)v17 != -1 )
    {
      if ( v45 == (const void *)-2LL )
      {
        v50 = (_QWORD *)((char *)a2 + 2) == 0;
      }
      else
      {
        if ( *(_QWORD *)(v17 + 8) != a3 )
        {
LABEL_55:
          if ( v22 || v45 != (const void *)-2LL )
            v17 = v22;
          v46 = v43 + v44;
          v22 = v17;
          ++v43;
          v44 = v41 & v46;
          continue;
        }
        v53 = v43;
        v56 = v22;
        v59 = v44;
        if ( !a3 )
          goto LABEL_41;
        v62 = v40 + 48LL * v44;
        v49 = memcmp(a2, v45, a3);
        v17 = v62;
        v44 = v59;
        v22 = v56;
        v43 = v53;
        v50 = v49 == 0;
      }
      if ( v50 )
        goto LABEL_41;
      if ( v45 == (const void *)-1LL )
        goto LABEL_39;
      goto LABEL_55;
    }
    break;
  }
LABEL_38:
  if ( a2 == (_QWORD *)-1LL )
    goto LABEL_41;
LABEL_39:
  if ( v22 )
    v17 = v22;
LABEL_41:
  v37 = *(_DWORD *)(a1 + 1456) + 1;
LABEL_42:
  *(_DWORD *)(a1 + 1456) = v37;
  if ( *(_QWORD *)v17 != -1 )
    --*(_DWORD *)(a1 + 1460);
  *(_QWORD *)(v17 + 8) = a3;
  *(_QWORD *)(v17 + 24) = 0;
  *(_QWORD *)v17 = a2;
  *(_QWORD *)(v17 + 16) = v17 + 32;
  *(_BYTE *)(v17 + 32) = 0;
  return v17 + 32;
}
