// Function: sub_21DEAC0
// Address: 0x21deac0
//
__int64 __fastcall sub_21DEAC0(__int64 a1, _QWORD *a2)
{
  int v4; // eax
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  _QWORD *v9; // r14
  unsigned int v10; // r10d
  __int64 v11; // r15
  _QWORD *v12; // rbx
  __int64 *v13; // rdx
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // r9
  __int64 v17; // r9
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rdi
  unsigned int v24; // eax
  int v25; // eax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  int v28; // ebx
  __int64 v29; // r13
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *j; // rdx
  _QWORD *v33; // rax
  _QWORD *v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 *v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  unsigned __int8 v39; // [rsp+28h] [rbp-48h]
  _DWORD v40[13]; // [rsp+3Ch] [rbp-34h] BYREF

  if ( !*(_DWORD *)(a2[1] + 952LL) )
    return sub_21DDFC0(a1, (__int64)a2);
  v4 = *(_DWORD *)(a1 + 248);
  ++*(_QWORD *)(a1 + 232);
  if ( !v4 )
  {
    if ( *(_DWORD *)(a1 + 252) )
    {
      v6 = *(unsigned int *)(a1 + 256);
      if ( (unsigned int)v6 <= 0x40 )
      {
LABEL_7:
        v7 = *(_QWORD **)(a1 + 240);
        for ( i = &v7[v6]; i != v7; ++v7 )
          *v7 = -8;
        *(_QWORD *)(a1 + 248) = 0;
        goto LABEL_10;
      }
      j___libc_free_0(*(_QWORD *)(a1 + 240));
      *(_QWORD *)(a1 + 240) = 0;
      *(_QWORD *)(a1 + 248) = 0;
      *(_DWORD *)(a1 + 256) = 0;
    }
LABEL_10:
    v9 = (_QWORD *)a2[41];
    v34 = a2 + 40;
    if ( a2 + 40 != v9 )
      goto LABEL_11;
    return 0;
  }
  v5 = 4 * v4;
  v6 = *(unsigned int *)(a1 + 256);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v5 = 64;
  if ( v5 >= (unsigned int)v6 )
    goto LABEL_7;
  v23 = *(_QWORD **)(a1 + 240);
  v24 = v4 - 1;
  if ( !v24 )
  {
    v29 = 1024;
    v28 = 128;
LABEL_64:
    j___libc_free_0(v23);
    *(_DWORD *)(a1 + 256) = v28;
    v30 = (_QWORD *)sub_22077B0(v29);
    v31 = *(unsigned int *)(a1 + 256);
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 240) = v30;
    for ( j = &v30[v31]; j != v30; ++v30 )
    {
      if ( v30 )
        *v30 = -8;
    }
    goto LABEL_10;
  }
  _BitScanReverse(&v24, v24);
  v25 = 1 << (33 - (v24 ^ 0x1F));
  if ( v25 < 64 )
    v25 = 64;
  if ( (_DWORD)v6 != v25 )
  {
    v26 = (4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1);
    v27 = ((v26 | (v26 >> 2)) >> 4) | v26 | (v26 >> 2) | ((((v26 | (v26 >> 2)) >> 4) | v26 | (v26 >> 2)) >> 8);
    v28 = (v27 | (v27 >> 16)) + 1;
    v29 = 8 * ((v27 | (v27 >> 16)) + 1);
    goto LABEL_64;
  }
  *(_QWORD *)(a1 + 248) = 0;
  v33 = &v23[v6];
  do
  {
    if ( v23 )
      *v23 = -8;
    ++v23;
  }
  while ( v33 != v23 );
  v9 = (_QWORD *)a2[41];
  v34 = a2 + 40;
  if ( v9 != a2 + 40 )
  {
LABEL_11:
    v10 = 0;
    while ( 1 )
    {
      v11 = v9[4];
      v12 = v9 + 3;
      if ( (_QWORD *)v11 != v9 + 3 )
        break;
LABEL_21:
      v9 = (_QWORD *)v9[1];
      if ( v9 == v34 )
        goto LABEL_22;
    }
    while ( 1 )
    {
      v13 = *(__int64 **)(*(_QWORD *)(v11 + 24) + 56LL);
      v14 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 16LL);
      if ( (v14 & 0x80u) != 0LL )
      {
        v35 = *(_QWORD *)(v11 + 16);
        v36 = *(__int64 **)(*(_QWORD *)(v11 + 24) + 56LL);
        v37 = *(_QWORD *)(v11 + 32) + 160LL;
        v15 = sub_21DC9A0(a1, v37, v13, v40);
        v13 = v36;
        v16 = v35;
        if ( v15 )
        {
          sub_1E313C0(v37, v40[0]);
          v16 = v35;
          v13 = v36;
        }
        v10 = 1;
        if ( (*(_BYTE *)(v16 + 17) & 0x10) != 0 )
          goto LABEL_19;
        v17 = *(_QWORD *)(v11 + 32) + 200LL;
        goto LABEL_34;
      }
      if ( (v14 & 0x300) != 0 )
        break;
      if ( (v14 & 0x400) != 0 )
      {
        v17 = *(_QWORD *)(v11 + 32);
        goto LABEL_34;
      }
      if ( (v14 & 0x800) != 0 )
      {
        v17 = *(_QWORD *)(v11 + 32) + 40LL;
        goto LABEL_34;
      }
LABEL_19:
      if ( (*(_BYTE *)v11 & 4) != 0 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v12 == (_QWORD *)v11 )
          goto LABEL_21;
      }
      else
      {
        while ( (*(_BYTE *)(v11 + 46) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v12 == (_QWORD *)v11 )
          goto LABEL_21;
      }
    }
    v17 = *(_QWORD *)(v11 + 32) + 40LL * (unsigned int)(1 << ((BYTE1(v14) & 3) - 1));
LABEL_34:
    v38 = v17;
    if ( (unsigned __int8)sub_21DC9A0(a1, v17, v13, v40) )
      sub_1E313C0(v38, v40[0]);
    v10 = 1;
    goto LABEL_19;
  }
  v10 = 0;
LABEL_22:
  if ( *(_DWORD *)(a1 + 248) )
  {
    v18 = *(__int64 **)(a1 + 240);
    v19 = &v18[*(unsigned int *)(a1 + 256)];
    if ( v18 != v19 )
    {
      while ( *v18 == -16 || *v18 == -8 )
      {
        if ( ++v18 == v19 )
          return v10;
      }
LABEL_45:
      if ( v18 != v19 )
      {
        v20 = a2[5];
        v21 = *(unsigned int *)(*(_QWORD *)(*v18 + 32) + 8LL);
        if ( (int)v21 < 0 )
          v22 = *(_QWORD *)(*(_QWORD *)(v20 + 24) + 16 * (v21 & 0x7FFFFFFF) + 8);
        else
          v22 = *(_QWORD *)(*(_QWORD *)(v20 + 272) + 8 * v21);
        while ( 1 )
        {
          if ( !v22 )
          {
            v39 = v10;
            sub_1E16240(*v18);
            v10 = v39;
            break;
          }
          if ( (*(_BYTE *)(v22 + 3) & 0x10) == 0 && (*(_BYTE *)(v22 + 4) & 8) == 0 )
            break;
          v22 = *(_QWORD *)(v22 + 32);
        }
        while ( ++v18 != v19 )
        {
          if ( *v18 != -16 && *v18 != -8 )
            goto LABEL_45;
        }
      }
    }
  }
  return v10;
}
