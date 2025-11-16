// Function: sub_D3B9E0
// Address: 0xd3b9e0
//
__int64 __fastcall sub_D3B9E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v9; // r14d
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 v12; // r9
  __int64 v13; // r8
  int v14; // r11d
  unsigned int v15; // edx
  unsigned int v16; // eax
  __int64 *v17; // r14
  unsigned __int64 v18; // r13
  unsigned int i; // ecx
  __int64 *v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // ecx
  int v23; // eax
  int v24; // edx
  __int64 v25; // r13
  __int64 v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rdi
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  int v39; // ecx
  int v40; // ecx
  __int64 v41; // rdx
  int v42; // edi
  __int64 *v43; // rsi
  unsigned int j; // eax
  __int64 v45; // r8
  unsigned int v46; // eax
  int v47; // eax
  int v48; // eax
  __int64 v49; // rdx
  int v50; // esi
  unsigned int v51; // r13d
  __int64 *v52; // rcx
  __int64 v53; // rdi
  unsigned int v54; // r13d
  __int64 v55; // [rsp+8h] [rbp-78h]
  __int64 v56; // [rsp+8h] [rbp-78h]
  __int64 *v57; // [rsp+10h] [rbp-70h]
  __int64 v58; // [rsp+10h] [rbp-70h]
  __int64 v62; // [rsp+28h] [rbp-58h]
  _QWORD v63[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v64[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( !a6 )
    goto LABEL_19;
  v9 = a3;
  v58 = sub_D970F0(a5);
  v10 = sub_D970F0(a5);
  v11 = *(_DWORD *)(a6 + 24);
  v12 = v10;
  if ( !v11 )
  {
    ++*(_QWORD *)a6;
    goto LABEL_37;
  }
  v13 = *(_QWORD *)(a6 + 8);
  v14 = 1;
  v15 = v9 >> 9;
  v16 = v9 >> 4;
  v17 = 0;
  v18 = ((0xBF58476D1CE4E5B9LL
        * (v15 ^ v16 | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL
       * (v15 ^ v16 | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)));
  for ( i = (((0xBF58476D1CE4E5B9LL
             * (v15 ^ v16 | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (v15 ^ v16)))
          & (v11 - 1); ; i = (v11 - 1) & v22 )
  {
    v20 = (__int64 *)(v13 + 32LL * i);
    v21 = *v20;
    if ( a2 == *v20 && a3 == v20[1] )
      return v20[2];
    if ( v21 == -4096 )
      break;
    if ( v21 == -8192 && v20[1] == -8192 && !v17 )
      v17 = (__int64 *)(v13 + 32LL * i);
LABEL_10:
    v22 = v14 + i;
    ++v14;
  }
  if ( v20[1] != -4096 )
    goto LABEL_10;
  v23 = *(_DWORD *)(a6 + 16);
  if ( !v17 )
    v17 = (__int64 *)(v13 + 32LL * i);
  ++*(_QWORD *)a6;
  v24 = v23 + 1;
  if ( 4 * (v23 + 1) >= 3 * v11 )
  {
LABEL_37:
    v55 = v12;
    sub_D3B710(a6, 2 * v11);
    v39 = *(_DWORD *)(a6 + 24);
    if ( v39 )
    {
      v40 = v39 - 1;
      v12 = v55;
      v42 = 1;
      v43 = 0;
      for ( j = v40
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v40 & v46 )
      {
        v41 = *(_QWORD *)(a6 + 8);
        v17 = (__int64 *)(v41 + 32LL * j);
        v45 = *v17;
        if ( a2 == *v17 && a3 == v17[1] )
          break;
        if ( v45 == -4096 )
        {
          if ( v17[1] == -4096 )
          {
            if ( v43 )
              v17 = v43;
            v24 = *(_DWORD *)(a6 + 16) + 1;
            goto LABEL_16;
          }
        }
        else if ( v45 == -8192 && v17[1] == -8192 && !v43 )
        {
          v43 = (__int64 *)(v41 + 32LL * j);
        }
        v46 = v42 + j;
        ++v42;
      }
      goto LABEL_58;
    }
LABEL_69:
    ++*(_DWORD *)(a6 + 16);
    BUG();
  }
  if ( v11 - *(_DWORD *)(a6 + 20) - v24 <= v11 >> 3 )
  {
    v56 = v12;
    sub_D3B710(a6, v11);
    v47 = *(_DWORD *)(a6 + 24);
    if ( v47 )
    {
      v48 = v47 - 1;
      v12 = v56;
      v50 = 1;
      v51 = v48 & v18;
      v52 = 0;
      while ( 1 )
      {
        v49 = *(_QWORD *)(a6 + 8);
        v17 = (__int64 *)(v49 + 32LL * v51);
        v53 = *v17;
        if ( a2 == *v17 && a3 == v17[1] )
          break;
        if ( v53 == -4096 )
        {
          if ( v17[1] == -4096 )
          {
            if ( v52 )
              v17 = v52;
            v24 = *(_DWORD *)(a6 + 16) + 1;
            goto LABEL_16;
          }
        }
        else if ( v53 == -8192 && v17[1] == -8192 && !v52 )
        {
          v52 = (__int64 *)(v49 + 32LL * v51);
        }
        v54 = v50 + v51;
        ++v50;
        v51 = v48 & v54;
      }
LABEL_58:
      v24 = *(_DWORD *)(a6 + 16) + 1;
      goto LABEL_16;
    }
    goto LABEL_69;
  }
LABEL_16:
  *(_DWORD *)(a6 + 16) = v24;
  if ( *v17 != -4096 || v17[1] != -4096 )
    --*(_DWORD *)(a6 + 20);
  *v17 = a2;
  v17[3] = v12;
  v17[1] = a3;
  v17[2] = v58;
  v57 = v17 + 2;
LABEL_19:
  if ( (unsigned __int8)sub_DADE90(a5, a2, a1) )
  {
    v26 = a2;
    v25 = a2;
    goto LABEL_27;
  }
  if ( *(_WORD *)(a2 + 24) == 8 )
  {
    v25 = **(_QWORD **)(a2 + 32);
    v26 = sub_DD0540(a2, a4, a5);
    v30 = sub_D33D80((_QWORD *)a2, a5, v27, v28, v29);
    if ( *(_WORD *)(v30 + 24) )
    {
      v25 = sub_DCEE80(a5, v25, v26, 0);
      v26 = sub_DCE050(a5, **(_QWORD **)(a2 + 32), v26);
    }
    else
    {
      v31 = *(_QWORD *)(v30 + 32);
      v32 = *(_DWORD *)(v31 + 32);
      v33 = *(_QWORD *)(v31 + 24);
      if ( v32 > 0x40 )
        v33 = *(_QWORD *)(v33 + 8LL * ((v32 - 1) >> 6));
      if ( (v33 & (1LL << ((unsigned __int8)v32 - 1))) != 0 )
      {
        v34 = v25;
        v25 = v26;
        v26 = v34;
      }
    }
LABEL_27:
    v62 = sub_AA4E30(**(_QWORD **)(a1 + 32));
    v35 = sub_D95540(a2);
    v36 = sub_AE4570(v62, v35);
    v64[1] = sub_DCAE60(a5, v36, a3);
    v63[0] = v64;
    v64[0] = v26;
    v63[1] = 0x200000002LL;
    v37 = sub_DC7EB0(a5, v63, 0, 0);
    if ( (_QWORD *)v63[0] != v64 )
      _libc_free(v63[0], v63);
    if ( a6 )
    {
      *v57 = v25;
      v57[1] = v37;
    }
  }
  else
  {
    v25 = sub_D970F0(a5);
    sub_D970F0(a5);
  }
  return v25;
}
