// Function: sub_3707F80
// Address: 0x3707f80
//
__int64 __fastcall sub_3707F80(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int64 v6; // r12
  int v7; // r14d
  int v8; // eax
  unsigned int v9; // esi
  int v10; // r10d
  unsigned int v11; // edx
  __int64 v12; // r8
  __int64 v13; // rax
  int v15; // edx
  int v16; // edx
  int v17; // r9d
  unsigned int v18; // r14d
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // r14d
  int v22; // eax
  int v23; // edx
  int v24; // edx
  int v25; // edx
  int v26; // r9d
  unsigned int v27; // r14d
  __int64 v28; // rax
  char **v29; // rdi
  char *v30; // r14
  __int64 v31; // r9
  __int64 v32; // rax
  _QWORD *v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // rax
  int v38; // eax
  unsigned int v39; // edx
  unsigned int v40; // r14d
  int v41; // [rsp+Ch] [rbp-64h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  int v43; // [rsp+18h] [rbp-58h]
  int v44; // [rsp+18h] [rbp-58h]
  __int64 v45; // [rsp+20h] [rbp-50h]
  __int64 v46; // [rsp+20h] [rbp-50h]
  __int64 v47; // [rsp+20h] [rbp-50h]
  __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+28h] [rbp-48h]
  __int64 v50; // [rsp+28h] [rbp-48h]

  v6 = sub_370C3D0(
         a2,
         a3,
         *(_QWORD *)(a1 + 120),
         *(unsigned int *)(a1 + 128),
         *(_QWORD *)(a1 + 120),
         *(unsigned int *)(a1 + 128));
  v7 = v6;
  v45 = a1 + 40;
  v8 = sub_3707C10(a1);
  v9 = *(_DWORD *)(a1 + 64);
  v10 = v8;
  if ( v9 )
  {
    v41 = 1;
    v42 = 0;
    v11 = v6 & (v9 - 1);
    v49 = (unsigned int)v6;
    while ( 1 )
    {
      v6 = v49 | v6 & 0xFFFFFFFF00000000LL;
      v12 = *(_QWORD *)(a1 + 48) + 12LL * v11;
      v13 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == v6 )
      {
        if ( *(_DWORD *)(v12 + 8) <= 0xFFFu )
          goto LABEL_29;
        return *(unsigned int *)(v12 + 8);
      }
      if ( v13 == unk_504EE80 )
        break;
      if ( !v42 )
      {
        if ( qword_504EE78 != v13 )
          v12 = 0;
        v42 = v12;
      }
      v39 = v41 + v11;
      ++v41;
      v11 = (v9 - 1) & v39;
    }
    if ( v42 )
      v12 = v42;
    v22 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) >= 3 * v9 )
      goto LABEL_7;
    if ( v9 - *(_DWORD *)(a1 + 60) - v23 > v9 >> 3 )
      goto LABEL_26;
    v44 = v10;
    sub_3707D20(v45, v9);
    v24 = *(_DWORD *)(a1 + 64);
    v12 = 0;
    v10 = v44;
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = 1;
      v27 = v25 & v7;
      v19 = 0;
      while ( 1 )
      {
        v6 = v49 | v6 & 0xFFFFFFFF00000000LL;
        v12 = *(_QWORD *)(a1 + 48) + 12LL * v27;
        v28 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 == v6 )
          break;
        if ( unk_504EE80 == v28 )
          goto LABEL_23;
        if ( qword_504EE78 != v28 || v19 )
          v12 = v19;
        v40 = v26 + v27;
        v19 = v12;
        ++v26;
        v27 = v25 & v40;
      }
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 40);
    v49 = (unsigned int)v6;
LABEL_7:
    v43 = v10;
    sub_3707D20(v45, 2 * v9);
    v15 = *(_DWORD *)(a1 + 64);
    v12 = 0;
    v10 = v43;
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = 1;
      v18 = v16 & v7;
      v19 = 0;
      while ( 1 )
      {
        v6 = v49 | v6 & 0xFFFFFFFF00000000LL;
        v12 = *(_QWORD *)(a1 + 48) + 12LL * v18;
        v20 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 == v6 )
          break;
        if ( unk_504EE80 == v20 )
        {
LABEL_23:
          if ( v19 )
            v12 = v19;
          break;
        }
        if ( v19 || qword_504EE78 != v20 )
          v12 = v19;
        v21 = v17 + v18;
        v19 = v12;
        ++v17;
        v18 = v16 & v21;
      }
    }
  }
  v23 = *(_DWORD *)(a1 + 56) + 1;
LABEL_26:
  *(_DWORD *)(a1 + 56) = v23;
  if ( unk_504EE80 != *(_QWORD *)v12 )
    --*(_DWORD *)(a1 + 60);
  *(_DWORD *)(v12 + 8) = v10;
  v6 = v49 | v6 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)v12 = v6;
LABEL_29:
  v29 = *(char ***)(a1 + 8);
  v30 = *v29;
  v29[10] += a3;
  if ( v29[1] >= &v30[a3] && v30 )
  {
    *v29 = &v30[a3];
  }
  else
  {
    v47 = v12;
    v37 = sub_9D1E70((__int64)v29, a3, a3, 0);
    v12 = v47;
    v30 = (char *)v37;
  }
  v46 = v12;
  memcpy(v30, a2, a3);
  v12 = v46;
  if ( a3 )
  {
    if ( *(_DWORD *)(v46 + 8) <= 0xFFFu )
    {
      v38 = sub_3707C10(a1);
      v12 = v46;
      *(_DWORD *)(v46 + 8) = v38;
    }
    v32 = *(unsigned int *)(a1 + 80);
    if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
    {
      v48 = v12;
      sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v32 + 1, 0x10u, v12, v31);
      v32 = *(unsigned int *)(a1 + 80);
      v12 = v48;
    }
    v33 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 16 * v32);
    *v33 = v30;
    v33[1] = a3;
    v34 = *(unsigned int *)(a1 + 132);
    v35 = *(unsigned int *)(a1 + 128);
    ++*(_DWORD *)(a1 + 80);
    v36 = v49 | v6 & 0xFFFFFFFF00000000LL;
    if ( v35 + 1 > v34 )
    {
      v50 = v12;
      sub_C8D5F0(a1 + 120, (const void *)(a1 + 136), v35 + 1, 8u, v12, v31);
      v35 = *(unsigned int *)(a1 + 128);
      v12 = v50;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8 * v35) = v36;
    ++*(_DWORD *)(a1 + 128);
    return *(unsigned int *)(v12 + 8);
  }
  else
  {
    *(_DWORD *)(v46 + 8) = 7;
    return 7;
  }
}
