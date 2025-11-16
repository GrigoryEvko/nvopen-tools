// Function: sub_2206E30
// Address: 0x2206e30
//
__int64 __fastcall sub_2206E30(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax
  void *v7; // rdi
  __int64 *v8; // rbx
  __int64 *v9; // r12
  unsigned int v11; // ecx
  _DWORD *v12; // rdi
  unsigned int v13; // edx
  int v14; // edx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  int v17; // ebx
  __int64 v18; // r14
  _DWORD *v19; // rax
  __int64 v20; // rdx
  _DWORD *i; // rdx
  __int64 *v22; // rbx
  __int64 *v23; // r12
  __int64 v24; // r15
  __int64 v25; // rax
  unsigned __int8 v26; // r14
  unsigned int v27; // esi
  __int64 v28; // r9
  unsigned int v29; // edx
  _QWORD *v30; // rdi
  __int64 v31; // rcx
  int v32; // r11d
  _QWORD *v33; // r8
  int v34; // ecx
  int v35; // ecx
  int v36; // edi
  int v37; // edi
  __int64 v38; // r9
  unsigned int v39; // eax
  __int64 v40; // r11
  int v41; // esi
  _QWORD *v42; // rdx
  int v43; // r9d
  int v44; // r9d
  __int64 v45; // r10
  _QWORD *v46; // rdi
  unsigned int v47; // eax
  int v48; // edx
  __int64 v49; // rsi
  _DWORD *v50; // rax
  __int64 v51; // [rsp+8h] [rbp-48h]
  unsigned __int8 v52; // [rsp+17h] [rbp-39h]
  int v53; // [rsp+18h] [rbp-38h]
  unsigned int v54; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 232) = *(_QWORD *)(a2 + 40);
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v4 = 0;
  if ( v3 != sub_1D00B00 )
    v4 = v3();
  *(_QWORD *)(a1 + 240) = v4;
  v51 = a1 + 248;
  sub_2206C80(a1 + 248);
  sub_2206C80(a1 + 280);
  v5 = *(_DWORD *)(a1 + 328);
  ++*(_QWORD *)(a1 + 312);
  if ( !v5 )
  {
    if ( !*(_DWORD *)(a1 + 332) )
      goto LABEL_9;
    LODWORD(v6) = *(_DWORD *)(a1 + 336);
    if ( (unsigned int)v6 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 320));
      *(_QWORD *)(a1 + 320) = 0;
      *(_QWORD *)(a1 + 328) = 0;
      *(_DWORD *)(a1 + 336) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v11 = 4 * v5;
  v6 = *(unsigned int *)(a1 + 336);
  if ( (unsigned int)(4 * v5) < 0x40 )
    v11 = 64;
  if ( v11 >= (unsigned int)v6 )
  {
LABEL_6:
    v7 = *(void **)(a1 + 320);
    if ( 4LL * (unsigned int)v6 )
      memset(v7, 255, 4LL * (unsigned int)v6);
    *(_QWORD *)(a1 + 328) = 0;
    goto LABEL_9;
  }
  v12 = *(_DWORD **)(a1 + 320);
  v13 = v5 - 1;
  if ( !v13 )
  {
    v18 = 512;
    v17 = 128;
LABEL_20:
    j___libc_free_0(v12);
    *(_DWORD *)(a1 + 336) = v17;
    v19 = (_DWORD *)sub_22077B0(v18);
    v20 = *(unsigned int *)(a1 + 336);
    *(_QWORD *)(a1 + 328) = 0;
    *(_QWORD *)(a1 + 320) = v19;
    for ( i = &v19[v20]; i != v19; ++v19 )
    {
      if ( v19 )
        *v19 = -1;
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v13, v13);
  v14 = 1 << (33 - (v13 ^ 0x1F));
  if ( v14 < 64 )
    v14 = 64;
  if ( (_DWORD)v6 != v14 )
  {
    v15 = (4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1);
    v16 = ((v15 | (v15 >> 2)) >> 4) | v15 | (v15 >> 2) | ((((v15 | (v15 >> 2)) >> 4) | v15 | (v15 >> 2)) >> 8);
    v17 = (v16 | (v16 >> 16)) + 1;
    v18 = 4 * ((v16 | (v16 >> 16)) + 1);
    goto LABEL_20;
  }
  *(_QWORD *)(a1 + 328) = 0;
  v50 = &v12[v6];
  do
  {
    if ( v12 )
      *v12 = -1;
    ++v12;
  }
  while ( v50 != v12 );
LABEL_9:
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 184LL))(a1, a2);
  v52 = 0;
  if ( *(_DWORD *)(a1 + 296) )
  {
    v22 = *(__int64 **)(a1 + 288);
    v23 = &v22[*(unsigned int *)(a1 + 304)];
    if ( v22 != v23 )
    {
      while ( *v22 == -8 || *v22 == -16 )
      {
        if ( v23 == ++v22 )
        {
          v52 = 0;
          goto LABEL_10;
        }
      }
      v52 = 0;
      if ( v23 == v22 )
        goto LABEL_10;
      while ( 1 )
      {
        v24 = *v22;
        v53 = *(_DWORD *)(*(_QWORD *)(*v22 + 32) + 48LL);
        v25 = sub_1E69D00(*(_QWORD *)(a1 + 232), v53);
        v26 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 192LL))(a1, v25);
        if ( !v26 )
          goto LABEL_42;
        sub_1E69C40(*(_QWORD *)(a1 + 232), *(_DWORD *)(*(_QWORD *)(v24 + 32) + 8LL), v53);
        v27 = *(_DWORD *)(a1 + 272);
        if ( !v27 )
          break;
        v28 = *(_QWORD *)(a1 + 256);
        v29 = (v27 - 1) & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
        v30 = (_QWORD *)(v28 + 8LL * v29);
        v31 = *v30;
        if ( v24 != *v30 )
        {
          v32 = 1;
          v33 = 0;
          while ( v31 != -8 )
          {
            if ( v33 || v31 != -16 )
              v30 = v33;
            v29 = (v27 - 1) & (v32 + v29);
            v31 = *(_QWORD *)(v28 + 8LL * v29);
            if ( v24 == v31 )
              goto LABEL_51;
            ++v32;
            v33 = v30;
            v30 = (_QWORD *)(v28 + 8LL * v29);
          }
          v34 = *(_DWORD *)(a1 + 264);
          if ( !v33 )
            v33 = v30;
          ++*(_QWORD *)(a1 + 248);
          v35 = v34 + 1;
          if ( 4 * v35 < 3 * v27 )
          {
            if ( v27 - *(_DWORD *)(a1 + 268) - v35 <= v27 >> 3 )
            {
              v54 = ((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9);
              sub_1E22DE0(v51, v27);
              v43 = *(_DWORD *)(a1 + 272);
              if ( !v43 )
              {
LABEL_97:
                ++*(_DWORD *)(a1 + 264);
                BUG();
              }
              v44 = v43 - 1;
              v45 = *(_QWORD *)(a1 + 256);
              v46 = 0;
              v47 = v44 & v54;
              v35 = *(_DWORD *)(a1 + 264) + 1;
              v33 = (_QWORD *)(v45 + 8LL * (v44 & v54));
              v48 = 1;
              v49 = *v33;
              if ( v24 != *v33 )
              {
                while ( v49 != -8 )
                {
                  if ( !v46 && v49 == -16 )
                    v46 = v33;
                  v47 = v44 & (v47 + v48);
                  v33 = (_QWORD *)(v45 + 8LL * v47);
                  v49 = *v33;
                  if ( v24 == *v33 )
                    goto LABEL_59;
                  ++v48;
                }
                if ( v46 )
                  v33 = v46;
              }
            }
            goto LABEL_59;
          }
LABEL_63:
          sub_1E22DE0(v51, 2 * v27);
          v36 = *(_DWORD *)(a1 + 272);
          if ( !v36 )
            goto LABEL_97;
          v37 = v36 - 1;
          v38 = *(_QWORD *)(a1 + 256);
          v39 = v37 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v35 = *(_DWORD *)(a1 + 264) + 1;
          v33 = (_QWORD *)(v38 + 8LL * v39);
          v40 = *v33;
          if ( v24 != *v33 )
          {
            v41 = 1;
            v42 = 0;
            while ( v40 != -8 )
            {
              if ( !v42 && v40 == -16 )
                v42 = v33;
              v39 = v37 & (v39 + v41);
              v33 = (_QWORD *)(v38 + 8LL * v39);
              v40 = *v33;
              if ( v24 == *v33 )
                goto LABEL_59;
              ++v41;
            }
            if ( v42 )
              v33 = v42;
          }
LABEL_59:
          *(_DWORD *)(a1 + 264) = v35;
          if ( *v33 != -8 )
            --*(_DWORD *)(a1 + 268);
          *v33 = v24;
        }
LABEL_51:
        v52 = v26;
LABEL_42:
        if ( ++v22 != v23 )
        {
          while ( *v22 == -16 || *v22 == -8 )
          {
            if ( v23 == ++v22 )
              goto LABEL_10;
          }
          if ( v23 != v22 )
            continue;
        }
        goto LABEL_10;
      }
      ++*(_QWORD *)(a1 + 248);
      goto LABEL_63;
    }
  }
LABEL_10:
  v8 = *(__int64 **)(a1 + 256);
  v9 = &v8[*(unsigned int *)(a1 + 272)];
  if ( *(_DWORD *)(a1 + 264) && v8 != v9 )
  {
    while ( *v8 == -8 || *v8 == -16 )
    {
      if ( ++v8 == v9 )
        return v52;
    }
LABEL_30:
    if ( v8 != v9 )
    {
      sub_1E16240(*v8);
      while ( ++v8 != v9 )
      {
        if ( *v8 != -16 && *v8 != -8 )
          goto LABEL_30;
      }
    }
  }
  return v52;
}
