// Function: sub_19F6B20
// Address: 0x19f6b20
//
void __fastcall sub_19F6B20(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 *v8; // rbx
  unsigned int v9; // r11d
  unsigned int v10; // r8d
  __int64 v11; // rdi
  unsigned int v12; // r10d
  __int64 *v13; // rax
  __int64 v14; // r9
  unsigned int v15; // r9d
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r10
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 *v21; // r13
  int v22; // edx
  int v23; // edx
  __int64 v24; // r8
  unsigned int v25; // edi
  int v26; // eax
  __int64 *v27; // rcx
  __int64 v28; // rsi
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // ecx
  int v33; // eax
  __int64 *v34; // r9
  __int64 v35; // rdi
  int v36; // eax
  int v37; // eax
  int v38; // edx
  int v39; // edx
  int v40; // r10d
  __int64 *v41; // r9
  __int64 v42; // r8
  unsigned int v43; // edi
  __int64 v44; // rsi
  int v45; // ebx
  __int64 *v46; // r11
  int v47; // r10d
  int v48; // [rsp+8h] [rbp-78h]
  unsigned int v49; // [rsp+10h] [rbp-70h]
  __int64 *v50; // [rsp+10h] [rbp-70h]
  int v51; // [rsp+10h] [rbp-70h]
  unsigned int v52; // [rsp+10h] [rbp-70h]
  __int64 v54; // [rsp+28h] [rbp-58h]
  __int64 v56; // [rsp+38h] [rbp-48h] BYREF
  __int64 v57; // [rsp+40h] [rbp-40h] BYREF
  __int64 *v58; // [rsp+48h] [rbp-38h] BYREF

  v56 = a3;
  if ( a1 == a2 || a1 + 1 == a2 )
    return;
  v3 = a1 + 1;
  do
  {
    while ( sub_19F52B0(&v56, *v3, *a1) )
    {
      v4 = *v3;
      v5 = v3 + 1;
      if ( a1 != v3 )
        memmove(a1 + 1, a1, (char *)v3 - (char *)a1);
      ++v3;
      *a1 = v4;
      if ( a2 == v5 )
        return;
    }
    v6 = *v3;
    v7 = v56;
    v8 = v3;
    v54 = v56 + 1400;
    v9 = ((unsigned int)*v3 >> 9) ^ ((unsigned int)*v3 >> 4);
    while ( 1 )
    {
      v19 = *(v8 - 1);
      v20 = *(_DWORD *)(v7 + 1424);
      v21 = v8;
      v57 = v19;
      if ( v20 )
      {
        v10 = v20 - 1;
        v11 = *(_QWORD *)(v7 + 1408);
        v12 = (v20 - 1) & v9;
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v6 == *v13 )
        {
LABEL_10:
          v15 = *((_DWORD *)v13 + 2);
          goto LABEL_11;
        }
        v51 = 1;
        v27 = 0;
        while ( v14 != -8 )
        {
          if ( v14 == -16 && !v27 )
            v27 = v13;
          v12 = v10 & (v51 + v12);
          v13 = (__int64 *)(v11 + 16LL * v12);
          v14 = *v13;
          if ( v6 == *v13 )
            goto LABEL_10;
          ++v51;
        }
        if ( !v27 )
          v27 = v13;
        v37 = *(_DWORD *)(v7 + 1416);
        ++*(_QWORD *)(v7 + 1400);
        v26 = v37 + 1;
        if ( 4 * v26 < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(v7 + 1420) - v26 > v20 >> 3 )
            goto LABEL_18;
          v52 = v9;
          sub_19F5120(v54, v20);
          v38 = *(_DWORD *)(v7 + 1424);
          if ( !v38 )
          {
LABEL_83:
            ++*(_DWORD *)(v7 + 1416);
            BUG();
          }
          v9 = v52;
          v39 = v38 - 1;
          v40 = 1;
          v41 = 0;
          v42 = *(_QWORD *)(v7 + 1408);
          v43 = v39 & v52;
          v26 = *(_DWORD *)(v7 + 1416) + 1;
          v27 = (__int64 *)(v42 + 16LL * (v39 & v52));
          v44 = *v27;
          if ( v6 == *v27 )
            goto LABEL_18;
          while ( v44 != -8 )
          {
            if ( v44 == -16 && !v41 )
              v41 = v27;
            v43 = v39 & (v40 + v43);
            v27 = (__int64 *)(v42 + 16LL * v43);
            v44 = *v27;
            if ( v6 == *v27 )
              goto LABEL_18;
            ++v40;
          }
          goto LABEL_46;
        }
      }
      else
      {
        ++*(_QWORD *)(v7 + 1400);
      }
      v49 = v9;
      sub_19F5120(v54, 2 * v20);
      v22 = *(_DWORD *)(v7 + 1424);
      if ( !v22 )
        goto LABEL_83;
      v9 = v49;
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v7 + 1408);
      v25 = v23 & v49;
      v26 = *(_DWORD *)(v7 + 1416) + 1;
      v27 = (__int64 *)(v24 + 16LL * (v23 & v49));
      v28 = *v27;
      if ( v6 == *v27 )
        goto LABEL_18;
      v47 = 1;
      v41 = 0;
      while ( v28 != -8 )
      {
        if ( !v41 && v28 == -16 )
          v41 = v27;
        v25 = v23 & (v47 + v25);
        v27 = (__int64 *)(v24 + 16LL * v25);
        v28 = *v27;
        if ( v6 == *v27 )
          goto LABEL_18;
        ++v47;
      }
LABEL_46:
      if ( v41 )
        v27 = v41;
LABEL_18:
      *(_DWORD *)(v7 + 1416) = v26;
      if ( *v27 != -8 )
        --*(_DWORD *)(v7 + 1420);
      *v27 = v6;
      *((_DWORD *)v27 + 2) = 0;
      v20 = *(_DWORD *)(v7 + 1424);
      if ( !v20 )
      {
        ++*(_QWORD *)(v7 + 1400);
        goto LABEL_22;
      }
      v11 = *(_QWORD *)(v7 + 1408);
      v19 = v57;
      v10 = v20 - 1;
      v15 = 0;
LABEL_11:
      v16 = v10 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v17 = (__int64 *)(v11 + 16LL * v16);
      v18 = *v17;
      if ( *v17 != v19 )
        break;
LABEL_12:
      --v8;
      if ( *((_DWORD *)v17 + 2) <= v15 )
        goto LABEL_27;
      v8[1] = *v8;
    }
    v48 = 1;
    v50 = 0;
    while ( v18 != -8 )
    {
      if ( v18 == -16 )
      {
        if ( v50 )
          v17 = v50;
        v50 = v17;
      }
      v16 = v10 & (v48 + v16);
      v17 = (__int64 *)(v11 + 16LL * v16);
      v18 = *v17;
      if ( *v17 == v19 )
        goto LABEL_12;
      ++v48;
    }
    v34 = v50;
    if ( !v50 )
      v34 = v17;
    v36 = *(_DWORD *)(v7 + 1416);
    ++*(_QWORD *)(v7 + 1400);
    v33 = v36 + 1;
    if ( 4 * v33 < 3 * v20 )
    {
      if ( v20 - (v33 + *(_DWORD *)(v7 + 1420)) <= v20 >> 3 )
      {
        sub_19F5120(v54, v20);
        sub_19E6B80(v54, &v57, &v58);
        v34 = v58;
        v19 = v57;
        v33 = *(_DWORD *)(v7 + 1416) + 1;
      }
      goto LABEL_24;
    }
LABEL_22:
    sub_19F5120(v54, 2 * v20);
    v29 = *(_DWORD *)(v7 + 1424);
    if ( !v29 )
    {
      ++*(_DWORD *)(v7 + 1416);
      BUG();
    }
    v19 = v57;
    v30 = v29 - 1;
    v31 = *(_QWORD *)(v7 + 1408);
    v32 = v30 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
    v33 = *(_DWORD *)(v7 + 1416) + 1;
    v34 = (__int64 *)(v31 + 16LL * v32);
    v35 = *v34;
    if ( *v34 != v57 )
    {
      v45 = 1;
      v46 = 0;
      while ( v35 != -8 )
      {
        if ( !v46 && v35 == -16 )
          v46 = v34;
        v32 = v30 & (v45 + v32);
        v34 = (__int64 *)(v31 + 16LL * v32);
        v35 = *v34;
        if ( v57 == *v34 )
          goto LABEL_24;
        ++v45;
      }
      if ( v46 )
        v34 = v46;
    }
LABEL_24:
    *(_DWORD *)(v7 + 1416) = v33;
    if ( *v34 != -8 )
      --*(_DWORD *)(v7 + 1420);
    *v34 = v19;
    *((_DWORD *)v34 + 2) = 0;
LABEL_27:
    *v21 = v6;
    ++v3;
  }
  while ( a2 != v3 );
}
