// Function: sub_DB4FC0
// Address: 0xdb4fc0
//
__int64 __fastcall sub_DB4FC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // rsi
  __int64 v10; // rdi
  unsigned int v11; // eax
  int v13; // esi
  unsigned int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 *v17; // r10
  int v18; // r11d
  unsigned int v19; // eax
  __int64 *v20; // r12
  __int64 v21; // rcx
  unsigned int v22; // eax
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // ecx
  int v29; // r9d
  int v30; // eax
  int v31; // eax
  int v32; // esi
  __int64 *v33; // r9
  __int64 v34; // r8
  int v35; // r10d
  unsigned int v36; // eax
  int v37; // r10d
  __int64 *v38; // r9
  __int64 v39; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  __int64 v42; // [rsp+18h] [rbp-38h] BYREF
  unsigned int v43; // [rsp+20h] [rbp-30h]

  v6 = *(unsigned int *)(a2 + 640);
  v7 = *(_QWORD *)(a2 + 624);
  if ( !(_DWORD)v6 )
    goto LABEL_10;
  v8 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v9 = (__int64 *)(v7 + 24LL * v8);
  v10 = *v9;
  if ( a3 != *v9 )
  {
    v13 = 1;
    while ( v10 != -4096 )
    {
      v29 = v13 + 1;
      v8 = (v6 - 1) & (v13 + v8);
      v9 = (__int64 *)(v7 + 24LL * v8);
      v10 = *v9;
      if ( a3 == *v9 )
        goto LABEL_3;
      v13 = v29;
    }
LABEL_10:
    sub_DB58E0(&v39, a2, a3, v7);
    v41 = a3;
    v43 = v40;
    if ( v40 > 0x40 )
      sub_C43780((__int64)&v42, (const void **)&v39);
    else
      v42 = v39;
    v14 = *(_DWORD *)(a2 + 640);
    if ( v14 )
    {
      v15 = v41;
      v16 = *(_QWORD *)(a2 + 624);
      v17 = 0;
      v18 = 1;
      v19 = (v14 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v20 = (__int64 *)(v16 + 24LL * v19);
      v21 = *v20;
      if ( v41 == *v20 )
      {
LABEL_14:
        if ( v43 > 0x40 && v42 )
          j_j___libc_free_0_0(v42);
LABEL_17:
        v22 = *((_DWORD *)v20 + 4);
        *(_DWORD *)(a1 + 8) = v22;
        if ( v22 > 0x40 )
          sub_C43780(a1, (const void **)v20 + 1);
        else
          *(_QWORD *)a1 = v20[1];
        if ( v40 > 0x40 )
        {
          if ( v39 )
            j_j___libc_free_0_0(v39);
        }
        return a1;
      }
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v17 )
          v17 = v20;
        v19 = (v14 - 1) & (v18 + v19);
        v20 = (__int64 *)(v16 + 24LL * v19);
        v21 = *v20;
        if ( v41 == *v20 )
          goto LABEL_14;
        ++v18;
      }
      v30 = *(_DWORD *)(a2 + 632);
      if ( v17 )
        v20 = v17;
      ++*(_QWORD *)(a2 + 616);
      v28 = v30 + 1;
      if ( 4 * (v30 + 1) < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a2 + 636) - v28 > v14 >> 3 )
        {
LABEL_27:
          *(_DWORD *)(a2 + 632) = v28;
          if ( *v20 != -4096 )
            --*(_DWORD *)(a2 + 636);
          *v20 = v15;
          *((_DWORD *)v20 + 4) = v43;
          v20[1] = v42;
          goto LABEL_17;
        }
        sub_DB4D90(a2 + 616, v14);
        v31 = *(_DWORD *)(a2 + 640);
        if ( v31 )
        {
          v32 = v31 - 1;
          v33 = 0;
          v34 = *(_QWORD *)(a2 + 624);
          v35 = 1;
          v28 = *(_DWORD *)(a2 + 632) + 1;
          v36 = (v31 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v20 = (__int64 *)(v34 + 24LL * v36);
          v15 = *v20;
          if ( v41 != *v20 )
          {
            while ( v15 != -4096 )
            {
              if ( !v33 && v15 == -8192 )
                v33 = v20;
              v36 = v32 & (v35 + v36);
              v20 = (__int64 *)(v34 + 24LL * v36);
              v15 = *v20;
              if ( v41 == *v20 )
                goto LABEL_27;
              ++v35;
            }
            v15 = v41;
            if ( v33 )
              v20 = v33;
          }
          goto LABEL_27;
        }
LABEL_63:
        ++*(_DWORD *)(a2 + 632);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a2 + 616);
    }
    sub_DB4D90(a2 + 616, 2 * v14);
    v23 = *(_DWORD *)(a2 + 640);
    if ( v23 )
    {
      v15 = v41;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a2 + 624);
      v26 = (v23 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v20 = (__int64 *)(v25 + 24LL * v26);
      v27 = *v20;
      v28 = *(_DWORD *)(a2 + 632) + 1;
      if ( *v20 != v41 )
      {
        v37 = 1;
        v38 = 0;
        while ( v27 != -4096 )
        {
          if ( !v38 && v27 == -8192 )
            v38 = v20;
          v26 = v24 & (v37 + v26);
          v20 = (__int64 *)(v25 + 24LL * v26);
          v27 = *v20;
          if ( v41 == *v20 )
            goto LABEL_27;
          ++v37;
        }
        if ( v38 )
          v20 = v38;
      }
      goto LABEL_27;
    }
    goto LABEL_63;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v7 + 24 * v6) )
    goto LABEL_10;
  v11 = *((_DWORD *)v9 + 4);
  *(_DWORD *)(a1 + 8) = v11;
  if ( v11 <= 0x40 )
  {
    *(_QWORD *)a1 = v9[1];
    return a1;
  }
  sub_C43780(a1, (const void **)v9 + 1);
  return a1;
}
