// Function: sub_1BA9430
// Address: 0x1ba9430
//
__int64 __fastcall sub_1BA9430(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v4; // r15d
  __int64 v5; // rdx
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r12
  int v11; // eax
  __int64 v12; // r13
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 i; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // r15
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r13
  char *v25; // rbx
  __int64 v26; // rax
  __int64 *v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rsi
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // esi
  __int64 v35; // rcx
  __int64 v36; // r8
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // rdi
  int v40; // esi
  int v41; // edx
  int v42; // r8d
  __int64 *v43; // r10
  int v44; // r11d
  int v45; // edi
  int v46; // edx
  int v47; // r10d
  __int64 *v48; // r9
  int v49; // edi
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+20h] [rbp-60h]
  __int64 v55; // [rsp+38h] [rbp-48h] BYREF
  _QWORD v56[2]; // [rsp+40h] [rbp-40h] BYREF
  char v57; // [rsp+50h] [rbp-30h] BYREF

  v3 = a1;
  v4 = *(_DWORD *)(a1 + 104);
  v55 = a2;
  v5 = *(_QWORD *)(a1 + 88);
  if ( v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16LL * v4) )
        return v7[1];
    }
    else
    {
      v11 = 1;
      while ( v8 != -8 )
      {
        v42 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v11 = v42;
      }
    }
  }
  v50 = v3 + 80;
  if ( a2 == **(_QWORD **)(*(_QWORD *)v3 + 32LL) )
  {
    v9 = 0;
    sub_1BA9300(v50, &v55)[1] = 0;
    return v9;
  }
  v12 = *(_QWORD *)(a2 + 8);
  if ( !v12 )
  {
    v9 = 0;
    goto LABEL_33;
  }
  v13 = v5;
  while ( 1 )
  {
    v14 = sub_1648700(v12);
    if ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) <= 9u )
      break;
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
    {
      v5 = v13;
      v9 = 0;
      goto LABEL_33;
    }
  }
  v9 = 0;
  for ( i = a2; ; i = v55 )
  {
    v16 = sub_1BA99C0(v3, v14[5], i, a3);
    if ( !v16 )
    {
      v34 = *(_DWORD *)(v3 + 104);
      if ( v34 )
      {
        v35 = v55;
        v36 = *(_QWORD *)(v3 + 88);
        v37 = (v34 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
        v38 = (__int64 *)(v36 + 16LL * v37);
        v39 = *v38;
        if ( v55 == *v38 )
        {
LABEL_41:
          v38[1] = 0;
          return 0;
        }
        v43 = 0;
        v44 = 1;
        while ( v39 != -8 )
        {
          if ( !v43 && v39 == -16 )
            v43 = v38;
          v37 = (v34 - 1) & (v44 + v37);
          v38 = (__int64 *)(v36 + 16LL * v37);
          v39 = *v38;
          if ( v55 == *v38 )
            goto LABEL_41;
          ++v44;
        }
        v45 = *(_DWORD *)(v3 + 96);
        if ( v43 )
          v38 = v43;
        ++*(_QWORD *)(v3 + 80);
        v46 = v45 + 1;
        if ( 4 * (v45 + 1) < 3 * v34 )
        {
          if ( v34 - *(_DWORD *)(v3 + 100) - v46 > v34 >> 3 )
          {
LABEL_58:
            *(_DWORD *)(v3 + 96) = v46;
            if ( *v38 != -8 )
              --*(_DWORD *)(v3 + 100);
            *v38 = v35;
            v38[1] = 0;
            goto LABEL_41;
          }
LABEL_63:
          sub_1BA9170(v50, v34);
          sub_1BA0FE0(v50, &v55, v56);
          v38 = (__int64 *)v56[0];
          v35 = v55;
          v46 = *(_DWORD *)(v3 + 96) + 1;
          goto LABEL_58;
        }
      }
      else
      {
        ++*(_QWORD *)(v3 + 80);
      }
      v34 *= 2;
      goto LABEL_63;
    }
    if ( v9 )
      break;
    v9 = v16;
    do
    {
LABEL_27:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        goto LABEL_32;
LABEL_28:
      v14 = sub_1648700(v12);
    }
    while ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) > 9u );
  }
  v17 = *(_QWORD **)(v3 + 40);
  v56[0] = v9;
  v56[1] = v16;
  v18 = sub_22077B0(120);
  v21 = v18;
  if ( v18 )
  {
    *(_BYTE *)(v18 + 40) = 2;
    *(_QWORD *)(v18 + 48) = v18 + 64;
    v22 = v18 + 96;
    v53 = v12;
    v23 = 0;
    v24 = v18 + 40;
    *(_QWORD *)(v18 + 56) = 0x100000000LL;
    *(_QWORD *)(v18 + 88) = 0x200000000LL;
    v52 = v3;
    v25 = (char *)v56;
    *(_QWORD *)(v18 + 72) = 0;
    *(_QWORD *)(v18 + 80) = v18 + 96;
    v51 = v18 + 80;
    while ( 1 )
    {
      *(_QWORD *)(v22 + 8 * v23) = v9;
      ++*(_DWORD *)(v21 + 88);
      v26 = *(unsigned int *)(v9 + 16);
      if ( (unsigned int)v26 >= *(_DWORD *)(v9 + 20) )
      {
        sub_16CD150(v9 + 8, (const void *)(v9 + 24), 0, 8, v19, v20);
        v26 = *(unsigned int *)(v9 + 16);
      }
      v25 += 8;
      *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8 * v26) = v24;
      ++*(_DWORD *)(v9 + 16);
      if ( v25 == &v57 )
        break;
      v9 = *(_QWORD *)v25;
      v23 = *(unsigned int *)(v21 + 88);
      if ( (unsigned int)v23 >= *(_DWORD *)(v21 + 92) )
      {
        sub_16CD150(v51, (const void *)(v21 + 96), 0, 8, v19, v20);
        v23 = *(unsigned int *)(v21 + 88);
      }
      v22 = *(_QWORD *)(v21 + 80);
    }
    *(_QWORD *)(v21 + 8) = 0;
    *(_QWORD *)(v21 + 16) = 0;
    v12 = v53;
    *(_BYTE *)(v21 + 24) = 2;
    v3 = v52;
    *(_QWORD *)(v21 + 32) = 0;
    *(_QWORD *)v21 = &unk_49F7160;
    *(_BYTE *)(v21 + 112) = 27;
  }
  if ( *v17 )
  {
    v27 = (__int64 *)v17[1];
    *(_QWORD *)(v21 + 32) = *v17;
    v28 = *(_QWORD *)(v21 + 8);
    v29 = *v27;
    *(_QWORD *)(v21 + 16) = v27;
    v29 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v21 + 8) = v29 | v28 & 7;
    *(_QWORD *)(v29 + 8) = v21 + 8;
    *v27 = *v27 & 7 | (v21 + 8);
LABEL_26:
    v9 = v21 + 40;
    goto LABEL_27;
  }
  v9 = 0;
  if ( v21 )
    goto LABEL_26;
  v12 = *(_QWORD *)(v12 + 8);
  if ( v12 )
    goto LABEL_28;
LABEL_32:
  v5 = *(_QWORD *)(v3 + 88);
  v4 = *(_DWORD *)(v3 + 104);
LABEL_33:
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 80);
LABEL_43:
    v40 = 2 * v4;
    goto LABEL_44;
  }
  v30 = v55;
  v31 = (v4 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
  v32 = (__int64 *)(v5 + 16LL * v31);
  v33 = *v32;
  if ( *v32 == v55 )
    goto LABEL_35;
  v47 = 1;
  v48 = 0;
  while ( v33 != -8 )
  {
    if ( !v48 && v33 == -16 )
      v48 = v32;
    v31 = (v4 - 1) & (v47 + v31);
    v32 = (__int64 *)(v5 + 16LL * v31);
    v33 = *v32;
    if ( v55 == *v32 )
      goto LABEL_35;
    ++v47;
  }
  v49 = *(_DWORD *)(v3 + 96);
  if ( v48 )
    v32 = v48;
  ++*(_QWORD *)(v3 + 80);
  v41 = v49 + 1;
  if ( 4 * (v49 + 1) >= 3 * v4 )
    goto LABEL_43;
  if ( v4 - (v41 + *(_DWORD *)(v3 + 100)) <= v4 >> 3 )
  {
    v40 = v4;
LABEL_44:
    sub_1BA9170(v50, v40);
    sub_1BA0FE0(v50, &v55, v56);
    v32 = (__int64 *)v56[0];
    v30 = v55;
    v41 = *(_DWORD *)(v3 + 96) + 1;
  }
  *(_DWORD *)(v3 + 96) = v41;
  if ( *v32 != -8 )
    --*(_DWORD *)(v3 + 100);
  *v32 = v30;
  v32[1] = 0;
LABEL_35:
  v32[1] = v9;
  return v9;
}
