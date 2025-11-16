// Function: sub_3708450
// Address: 0x3708450
//
__int64 __fastcall sub_3708450(__int64 a1, int *a2, void *a3, size_t a4, char a5)
{
  __int64 v9; // rax
  unsigned int v10; // esi
  int v11; // r10d
  unsigned __int64 v12; // rbx
  int v13; // r8d
  __int64 v14; // rdx
  __int64 v15; // r11
  __int64 *v16; // rcx
  __int64 v17; // rax
  int v19; // edi
  int v20; // edi
  unsigned int i; // r8d
  __int64 v22; // rax
  unsigned int v23; // r8d
  int v24; // eax
  int v25; // ecx
  int v26; // edi
  int v27; // edi
  unsigned int j; // r8d
  __int64 v29; // rax
  unsigned __int64 v30; // rbx
  _QWORD *v31; // rax
  char **v32; // rdi
  char *v33; // rcx
  unsigned int v34; // r8d
  __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  __int64 v38; // [rsp+20h] [rbp-50h]
  int v39; // [rsp+20h] [rbp-50h]
  int v40; // [rsp+20h] [rbp-50h]
  unsigned int v41; // [rsp+28h] [rbp-48h]
  int v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+28h] [rbp-48h]
  int v44; // [rsp+28h] [rbp-48h]
  int v45; // [rsp+30h] [rbp-40h]
  int v46; // [rsp+30h] [rbp-40h]
  int v47; // [rsp+30h] [rbp-40h]
  void *v49; // [rsp+38h] [rbp-38h]

  v49 = a3;
  v9 = sub_370C3D0(
         a3,
         a4,
         *(_QWORD *)(a1 + 120),
         *(unsigned int *)(a1 + 128),
         *(_QWORD *)(a1 + 120),
         *(unsigned int *)(a1 + 128));
  v10 = *(_DWORD *)(a1 + 64);
  v11 = *a2;
  v12 = v9;
  v13 = v9;
  v37 = a1 + 40;
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 40);
    v15 = (unsigned int)v9;
LABEL_6:
    v35 = v15;
    v42 = v11;
    v46 = v13;
    sub_3707D20(v37, 2 * v10);
    v19 = *(_DWORD *)(a1 + 64);
    v14 = 0;
    v11 = v42;
    v15 = v35;
    if ( v19 )
    {
      v20 = v19 - 1;
      v39 = 1;
      v43 = 0;
      for ( i = v20 & v46; ; i = v20 & v23 )
      {
        v12 = v35 | v12 & 0xFFFFFFFF00000000LL;
        v14 = *(_QWORD *)(a1 + 48) + 12LL * i;
        v22 = *(_QWORD *)v14;
        if ( *(_QWORD *)v14 == v12 )
          break;
        if ( unk_504EE80 == v22 )
        {
LABEL_22:
          if ( v43 )
            v14 = v43;
          goto LABEL_24;
        }
        if ( v43 || qword_504EE78 != v22 )
          v14 = v43;
        v43 = v14;
        v23 = v39 + i;
        ++v39;
      }
    }
    goto LABEL_24;
  }
  v14 = 0;
  v15 = (unsigned int)v9;
  v45 = 1;
  v38 = *(_QWORD *)(a1 + 48);
  v41 = v9 & (v10 - 1);
  while ( 1 )
  {
    v12 = v15 | v12 & 0xFFFFFFFF00000000LL;
    v16 = (__int64 *)(v38 + 12LL * v41);
    v17 = *v16;
    if ( *v16 == v12 )
    {
      *a2 = *((_DWORD *)v16 + 2);
      return 0;
    }
    if ( v17 == unk_504EE80 )
      break;
    if ( !v14 && qword_504EE78 == v17 )
      v14 = v38 + 12LL * v41;
    v41 = (v10 - 1) & (v45 + v41);
    ++v45;
  }
  v24 = *(_DWORD *)(a1 + 56);
  if ( !v14 )
    v14 = v38 + 12LL * v41;
  ++*(_QWORD *)(a1 + 40);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v10 )
    goto LABEL_6;
  if ( v10 - *(_DWORD *)(a1 + 60) - v25 <= v10 >> 3 )
  {
    v36 = v15;
    v44 = v11;
    v47 = v13;
    sub_3707D20(v37, v10);
    v26 = *(_DWORD *)(a1 + 64);
    v14 = 0;
    v11 = v44;
    v15 = v36;
    if ( v26 )
    {
      v27 = v26 - 1;
      v40 = 1;
      v43 = 0;
      for ( j = v27 & v47; ; j = v27 & v34 )
      {
        v12 = v36 | v12 & 0xFFFFFFFF00000000LL;
        v14 = *(_QWORD *)(a1 + 48) + 12LL * j;
        v29 = *(_QWORD *)v14;
        if ( *(_QWORD *)v14 == v12 )
          break;
        if ( unk_504EE80 == v29 )
          goto LABEL_22;
        if ( qword_504EE78 != v29 || v43 )
          v14 = v43;
        v43 = v14;
        v34 = v40 + j;
        ++v40;
      }
    }
LABEL_24:
    v25 = *(_DWORD *)(a1 + 56) + 1;
  }
  *(_DWORD *)(a1 + 56) = v25;
  if ( unk_504EE80 != *(_QWORD *)v14 )
    --*(_DWORD *)(a1 + 60);
  v30 = v15 | v12 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)v14 = v30;
  *(_DWORD *)(v14 + 8) = (v11 & 0x7FFFFFFF) - 4096;
  if ( a5 )
  {
    v32 = *(char ***)(a1 + 8);
    v33 = *v32;
    v32[10] += a4;
    if ( v32[1] >= &v33[a4] && v33 )
      *v32 = &v33[a4];
    else
      v33 = (char *)sub_9D1E70((__int64)v32, a4, a4, 0);
    v49 = memcpy(v33, a3, a4);
  }
  v31 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL * ((*a2 & 0x7FFFFFFFu) - 4096));
  *v31 = v49;
  v31[1] = a4;
  *(_QWORD *)(*(_QWORD *)(a1 + 120) + 8LL * ((*a2 & 0x7FFFFFFFu) - 4096)) = v30;
  return 1;
}
