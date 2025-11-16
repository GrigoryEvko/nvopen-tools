// Function: sub_3931020
// Address: 0x3931020
//
__int64 __fastcall sub_3931020(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r10
  __int64 v9; // r8
  unsigned int v10; // r15d
  unsigned int v11; // edi
  __int64 *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  const char *v15; // r15
  char *v16; // rsi
  size_t v17; // r8
  size_t v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rcx
  char v21; // al
  __int64 v22; // r9
  int v23; // ecx
  int v24; // r8d
  int v25; // edx
  __int64 v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // r14
  int v30; // r11d
  __int64 *v31; // rdx
  int v32; // eax
  int v33; // ecx
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  unsigned int v37; // eax
  __int64 v38; // rdi
  int v39; // r10d
  __int64 *v40; // r9
  int v41; // eax
  int v42; // eax
  __int64 v43; // rdi
  __int64 *v44; // r8
  unsigned int v45; // r15d
  int v46; // r9d
  __int64 v47; // rsi
  size_t v48; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v49; // [rsp+10h] [rbp-70h] BYREF
  __int16 v50; // [rsp+20h] [rbp-60h]
  _QWORD *v51; // [rsp+30h] [rbp-50h] BYREF
  size_t v52; // [rsp+38h] [rbp-48h]
  _QWORD v53[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = *a1;
  v6 = *(_DWORD *)(*a1 + 40);
  v7 = *a1 + 16;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 16);
    goto LABEL_37;
  }
  v9 = *(_QWORD *)(v5 + 24);
  v10 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v11 = (v6 - 1) & v10;
  v12 = (__int64 *)(v9 + 32LL * v11);
  v13 = *v12;
  if ( a3 == *v12 )
    goto LABEL_3;
  v30 = 1;
  v31 = 0;
  while ( 1 )
  {
    if ( v13 == -8 )
    {
      if ( !v31 )
        v31 = v12;
      v32 = *(_DWORD *)(v5 + 32);
      ++*(_QWORD *)(v5 + 16);
      v33 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v5 + 36) - v33 > v6 >> 3 )
        {
LABEL_33:
          *(_DWORD *)(v5 + 32) = v33;
          if ( *v31 != -8 )
            --*(_DWORD *)(v5 + 36);
          *v31 = a3;
          v28 = 0;
          v31[1] = 0;
          v31[2] = 0;
          v31[3] = 0;
          return v28;
        }
        sub_3930E00(v7, v6);
        v41 = *(_DWORD *)(v5 + 40);
        if ( v41 )
        {
          v42 = v41 - 1;
          v43 = *(_QWORD *)(v5 + 24);
          v44 = 0;
          v45 = v42 & v10;
          v46 = 1;
          v33 = *(_DWORD *)(v5 + 32) + 1;
          v31 = (__int64 *)(v43 + 32LL * v45);
          v47 = *v31;
          if ( a3 != *v31 )
          {
            while ( v47 != -8 )
            {
              if ( v47 == -16 && !v44 )
                v44 = v31;
              v45 = v42 & (v46 + v45);
              v31 = (__int64 *)(v43 + 32LL * v45);
              v47 = *v31;
              if ( a3 == *v31 )
                goto LABEL_33;
              ++v46;
            }
            if ( v44 )
              v31 = v44;
          }
          goto LABEL_33;
        }
LABEL_70:
        ++*(_DWORD *)(v5 + 32);
        BUG();
      }
LABEL_37:
      sub_3930E00(v7, 2 * v6);
      v34 = *(_DWORD *)(v5 + 40);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(v5 + 24);
        v37 = (v34 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v33 = *(_DWORD *)(v5 + 32) + 1;
        v31 = (__int64 *)(v36 + 32LL * v37);
        v38 = *v31;
        if ( a3 != *v31 )
        {
          v39 = 1;
          v40 = 0;
          while ( v38 != -8 )
          {
            if ( v38 == -16 && !v40 )
              v40 = v31;
            v37 = v35 & (v39 + v37);
            v31 = (__int64 *)(v36 + 32LL * v37);
            v38 = *v31;
            if ( a3 == *v31 )
              goto LABEL_33;
            ++v39;
          }
          if ( v40 )
            v31 = v40;
        }
        goto LABEL_33;
      }
      goto LABEL_70;
    }
    if ( v13 != -16 || v31 )
      v12 = v31;
    v11 = (v6 - 1) & (v30 + v11);
    v13 = *(_QWORD *)(v9 + 32LL * v11);
    if ( a3 == v13 )
      break;
    ++v30;
    v31 = v12;
    v12 = (__int64 *)(v9 + 32LL * v11);
  }
  v12 = (__int64 *)(v9 + 32LL * v11);
LABEL_3:
  if ( v12[1] == v12[2] )
    return 0;
  v14 = *(_QWORD *)(v5 + 8);
  v15 = ".rela";
  v16 = *(char **)(a3 + 152);
  v17 = *(_QWORD *)(a3 + 160);
  v51 = v53;
  if ( (*(_BYTE *)(v14 + 12) & 1) == 0 )
    v15 = ".rel";
  v48 = v17;
  v18 = strlen(v15);
  if ( (unsigned int)v18 < 8 )
  {
    if ( (v18 & 4) != 0 )
    {
      LODWORD(v53[0]) = *(_DWORD *)v15;
      *(_DWORD *)((char *)&v52 + (unsigned int)v18 + 4) = *(_DWORD *)&v15[(unsigned int)v18 - 4];
    }
    else if ( (_DWORD)v18 )
    {
      LOBYTE(v53[0]) = *v15;
      if ( (v18 & 2) != 0 )
        *(_WORD *)((char *)&v52 + (unsigned int)v18 + 6) = *(_WORD *)&v15[(unsigned int)v18 - 2];
    }
  }
  else
  {
    *(_QWORD *)((char *)&v53[-1] + (unsigned int)v18) = *(_QWORD *)&v15[(unsigned int)v18 - 8];
    if ( (unsigned int)(v18 - 1) >= 8 )
    {
      v19 = 0;
      do
      {
        v20 = v19;
        v19 += 8;
        *(_QWORD *)((char *)v53 + v20) = *(_QWORD *)&v15[v20];
      }
      while ( v19 < (((_DWORD)v18 - 1) & 0xFFFFFFF8) );
    }
  }
  v52 = v18;
  *((_BYTE *)v53 + v18) = 0;
  if ( v48 > 0x3FFFFFFFFFFFFFFFLL - v52 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v51, v16, v48);
  v21 = *(_BYTE *)(*(_QWORD *)(*a1 + 8) + 12LL);
  if ( (v21 & 1) != 0 )
  {
    v22 = *(_QWORD *)(a3 + 184);
    v23 = *(_DWORD *)(a3 + 172) & 0x200;
    if ( (v21 & 2) != 0 )
    {
      v24 = 24;
      if ( v23 )
        v23 = 512;
LABEL_15:
      v25 = 4;
    }
    else
    {
      if ( !v23 )
      {
        v24 = 12;
        goto LABEL_15;
      }
      v23 = 512;
      v24 = 12;
      v25 = 4;
    }
  }
  else
  {
    v25 = 9;
    v22 = *(_QWORD *)(a3 + 184);
    v24 = (v21 & 2) == 0 ? 8 : 16;
    v23 = *(_DWORD *)(a3 + 172) & 0x200;
    if ( v23 )
      v23 = 512;
  }
  v49 = (unsigned __int64 *)&v51;
  v50 = 260;
  v26 = sub_38BEB70(a2, (__int64)&v49, v25, v23, v24, v22, a3);
  v27 = v51;
  v28 = v26;
  *(_DWORD *)(v26 + 24) = (*(_BYTE *)(*(_QWORD *)(*a1 + 8) + 12LL) & 2) == 0 ? 4 : 8;
  if ( v27 != v53 )
    j_j___libc_free_0((unsigned __int64)v27);
  return v28;
}
