// Function: sub_C2D850
// Address: 0xc2d850
//
_BYTE *__fastcall sub_C2D850(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 *v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r9
  unsigned int v16; // ecx
  __int64 *v17; // rdi
  __int64 v18; // r8
  __int64 *v19; // rdx
  int v20; // ecx
  int v21; // ecx
  int v22; // ecx
  int v23; // r11d
  __int64 v24; // r10
  unsigned int v25; // esi
  __int64 v26; // r9
  int v27; // r8d
  __int64 *v28; // rdi
  int v29; // edx
  int v30; // r11d
  __int64 v31; // r10
  int v32; // r8d
  unsigned int v33; // esi
  __int64 v34; // r9
  __int64 v35; // rdi
  char *v36; // rcx
  char *(*v37)(); // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // [rsp+10h] [rbp-B0h]
  __int64 v41; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v42; // [rsp+20h] [rbp-A0h]
  __int64 v43; // [rsp+20h] [rbp-A0h]
  _QWORD *v44; // [rsp+28h] [rbp-98h]
  const char *v45; // [rsp+30h] [rbp-90h] BYREF
  char v46; // [rsp+50h] [rbp-70h]
  char v47; // [rsp+51h] [rbp-6Fh]
  void *v48; // [rsp+60h] [rbp-60h] BYREF
  __int64 v49; // [rsp+68h] [rbp-58h]
  char *v50; // [rsp+70h] [rbp-50h]
  __int64 v51; // [rsp+78h] [rbp-48h]
  int v52; // [rsp+80h] [rbp-40h]
  const char **v53; // [rsp+88h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 48);
  if ( !*(_BYTE *)(v2 + 204) )
  {
    result = *(_BYTE **)(v2 + 24);
    v44 = result;
    if ( !result )
    {
LABEL_38:
      *(_BYTE *)(a1 + 56) = 1;
      return result;
    }
    while ( 1 )
    {
      v48 = 0;
      v49 = 0;
      v50 = 0;
      v51 = 0;
      sub_C1E3C0(v44 + 2, (__int64)&v48);
      if ( (_DWORD)v50 )
        break;
LABEL_4:
      v5 = (unsigned int)v51;
      v6 = v49;
LABEL_5:
      sub_C7D6A0(v6, 16 * v5, 8);
      result = (_BYTE *)*v44;
      v44 = result;
      if ( !result )
        goto LABEL_38;
    }
    v5 = (unsigned int)v51;
    v6 = v49;
    v7 = (__int64 *)v49;
    v8 = (__int64 *)(v49 + 16LL * (unsigned int)v51);
    if ( (__int64 *)v49 == v8 )
      goto LABEL_5;
    while ( 1 )
    {
      v9 = v7[1];
      if ( (v9 != -1 || *v7) && (v9 != -2 || *v7) )
        break;
      v7 += 2;
      if ( v8 == v7 )
        goto LABEL_5;
    }
    if ( v8 == v7 )
      goto LABEL_5;
    while ( 1 )
    {
      v10 = *v7;
      v11 = 0;
      if ( *v7 )
        v11 = v7[1];
      v12 = sub_EF75F0(*(_QWORD *)(a1 + 8), *v7, v11);
      if ( v12 )
        break;
LABEL_20:
      v7 += 2;
      if ( v7 != v8 )
      {
        while ( 1 )
        {
          v13 = v7[1];
          if ( (v13 != -1 || *v7) && (v13 != -2 || *v7) )
            break;
          v7 += 2;
          if ( v8 == v7 )
            goto LABEL_4;
        }
        if ( v8 != v7 )
          continue;
      }
      goto LABEL_4;
    }
    v14 = *(_DWORD *)(a1 + 40);
    if ( v14 )
    {
      v15 = *(_QWORD *)(a1 + 24);
      v42 = ((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (0xBF58476D1CE4E5B9LL * v12);
      v16 = (v14 - 1) & (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12));
      v17 = (__int64 *)(v15 + 24LL * v16);
      v18 = *v17;
      if ( v12 == *v17 )
        goto LABEL_20;
      v40 = 1;
      v19 = 0;
      while ( v18 != -1 )
      {
        if ( !v19 && v18 == -2 )
          v19 = v17;
        v16 = (v14 - 1) & (v40 + v16);
        v17 = (__int64 *)(v15 + 24LL * v16);
        v18 = *v17;
        if ( v12 == *v17 )
          goto LABEL_20;
        ++v40;
      }
      v20 = *(_DWORD *)(a1 + 32);
      if ( !v19 )
        v19 = v17;
      ++*(_QWORD *)(a1 + 16);
      v21 = v20 + 1;
      if ( 4 * v21 < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a1 + 36) - v21 > v14 >> 3 )
          goto LABEL_35;
        v41 = v12;
        sub_9E2150(a1 + 16, v14);
        v29 = *(_DWORD *)(a1 + 40);
        if ( !v29 )
        {
LABEL_68:
          ++*(_DWORD *)(a1 + 32);
          BUG();
        }
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 24);
        v28 = 0;
        v32 = 1;
        v33 = (v29 - 1) & v42;
        v21 = *(_DWORD *)(a1 + 32) + 1;
        v12 = v41;
        v19 = (__int64 *)(v31 + 24LL * v33);
        v34 = *v19;
        if ( v41 == *v19 )
          goto LABEL_35;
        while ( v34 != -1 )
        {
          if ( !v28 && v34 == -2 )
            v28 = v19;
          v33 = v30 & (v32 + v33);
          v19 = (__int64 *)(v31 + 24LL * v33);
          v34 = *v19;
          if ( v41 == *v19 )
            goto LABEL_35;
          ++v32;
        }
        goto LABEL_52;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 16);
    }
    v43 = v12;
    sub_9E2150(a1 + 16, 2 * v14);
    v22 = *(_DWORD *)(a1 + 40);
    if ( !v22 )
      goto LABEL_68;
    v12 = v43;
    v23 = v22 - 1;
    v24 = *(_QWORD *)(a1 + 24);
    v21 = *(_DWORD *)(a1 + 32) + 1;
    v25 = v23 & (((0xBF58476D1CE4E5B9LL * v43) >> 31) ^ (484763065 * v43));
    v19 = (__int64 *)(v24 + 24LL * v25);
    v26 = *v19;
    if ( v43 == *v19 )
      goto LABEL_35;
    v27 = 1;
    v28 = 0;
    while ( v26 != -1 )
    {
      if ( !v28 && v26 == -2 )
        v28 = v19;
      v25 = v23 & (v27 + v25);
      v19 = (__int64 *)(v24 + 24LL * v25);
      v26 = *v19;
      if ( v43 == *v19 )
        goto LABEL_35;
      ++v27;
    }
LABEL_52:
    if ( v28 )
      v19 = v28;
LABEL_35:
    *(_DWORD *)(a1 + 32) = v21;
    if ( *v19 != -1 )
      --*(_DWORD *)(a1 + 36);
    *v19 = v12;
    v19[1] = v10;
    v19[2] = v11;
    goto LABEL_20;
  }
  v35 = *(_QWORD *)(v2 + 72);
  v47 = 1;
  v45 = "Profile data remapping cannot be applied to profile data using MD5 names (original mangled names are not available).";
  v36 = "Unknown buffer";
  v46 = 3;
  v37 = *(char *(**)())(*(_QWORD *)v35 + 16LL);
  v38 = 14;
  if ( v37 != sub_C1E8B0 )
  {
    v36 = (char *)((__int64 (__fastcall *)(__int64, __int64, char *(*)(), char *))v37)(v35, a2, v37, "Unknown buffer");
    v38 = v39;
  }
  v51 = v38;
  v49 = 0x10000000CLL;
  v50 = v36;
  v48 = &unk_49D9C78;
  v52 = 0;
  v53 = &v45;
  return sub_B6EB20(a2, (__int64)&v48);
}
