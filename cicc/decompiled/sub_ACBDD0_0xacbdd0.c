// Function: sub_ACBDD0
// Address: 0xacbdd0
//
__int64 __fastcall sub_ACBDD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 *v7; // rsi
  __int64 v8; // rdx
  int v9; // r10d
  __int64 *v10; // rcx
  unsigned int i; // eax
  __int64 *v12; // rdi
  __int64 v13; // r9
  unsigned int v14; // eax
  __int64 *v15; // r14
  __int64 v16; // r15
  __int64 v18; // r14
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rdx
  int v26; // r9d
  __int64 v27; // rdi
  int v28; // r9d
  int v29; // r10d
  unsigned int j; // eax
  _QWORD *v31; // r8
  unsigned int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  int v38; // edx
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 *v41; // [rsp+8h] [rbp-48h] BYREF
  __int64 v42; // [rsp+10h] [rbp-40h] BYREF
  __int64 v43; // [rsp+18h] [rbp-38h]

  v3 = a3;
  v5 = *(_QWORD *)(a1 - 64);
  if ( v5 == a2 )
  {
    v18 = *(_QWORD *)(a1 - 32);
    v5 = sub_BD3990(a3);
    v3 = v18;
  }
  v6 = *(_QWORD *)sub_BD5C60(a1, a2, a3);
  v42 = v5;
  v43 = v3;
  v7 = (__int64 *)*(unsigned int *)(v6 + 2016);
  if ( !(_DWORD)v7 )
  {
    ++*(_QWORD *)(v6 + 1992);
    v41 = 0;
    goto LABEL_51;
  }
  v8 = *(_QWORD *)(v6 + 2000);
  v9 = 1;
  v10 = 0;
  for ( i = ((_DWORD)v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)
              | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)))); ; i = ((_DWORD)v7 - 1) & v14 )
  {
    v12 = (__int64 *)(v8 + 24LL * i);
    v13 = *v12;
    if ( v5 == *v12 && v3 == v12[1] )
    {
      v15 = v12 + 2;
      goto LABEL_14;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v10 )
      v10 = (__int64 *)(v8 + 24LL * i);
LABEL_11:
    v14 = v9 + i;
    ++v9;
  }
  if ( v12[1] != -4096 )
    goto LABEL_11;
  v37 = *(_DWORD *)(v6 + 2008);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)(v6 + 1992);
  v38 = v37 + 1;
  v41 = v10;
  if ( 4 * (v37 + 1) < (unsigned int)(3 * (_DWORD)v7) )
  {
    v39 = v5;
    if ( (int)v7 - *(_DWORD *)(v6 + 2012) - v38 > (unsigned int)v7 >> 3 )
      goto LABEL_43;
    goto LABEL_52;
  }
LABEL_51:
  LODWORD(v7) = 2 * (_DWORD)v7;
LABEL_52:
  sub_ACBB00(v6 + 1992, (int)v7);
  v7 = &v42;
  sub_AC6F30(v6 + 1992, &v42, &v41);
  v39 = v42;
  v10 = v41;
  v38 = *(_DWORD *)(v6 + 2008) + 1;
LABEL_43:
  *(_DWORD *)(v6 + 2008) = v38;
  if ( *v10 != -4096 || v10[1] != -4096 )
    --*(_DWORD *)(v6 + 2012);
  *v10 = v39;
  v40 = v43;
  v15 = v10 + 2;
  v10[2] = 0;
  v10[1] = v40;
LABEL_14:
  v16 = *v15;
  if ( !*v15 )
  {
    v19 = *(_QWORD *)(a1 - 32);
    v20 = *(unsigned __int16 *)(v19 + 2);
    v21 = (unsigned int)(v20 + 0x7FFF);
    LOWORD(v21) = (v20 + 0x7FFF) & 0x7FFF;
    *(_WORD *)(v19 + 2) = v21 | v20 & 0x8000;
    v22 = sub_BD5C60(a1, v7, v21);
    v23 = *(_QWORD *)(a1 - 32);
    v24 = *(_QWORD *)(a1 - 64);
    v25 = *(_QWORD *)v22;
    v26 = *(_DWORD *)(*(_QWORD *)v22 + 2016LL);
    v27 = *(_QWORD *)(*(_QWORD *)v22 + 2000LL);
    if ( v26 )
    {
      v28 = v26 - 1;
      v29 = 1;
      for ( j = v28
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                  | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; j = v28 & v32 )
      {
        v31 = (_QWORD *)(v27 + 24LL * j);
        if ( v24 == *v31 && v23 == v31[1] )
          break;
        if ( *v31 == -4096 && v31[1] == -4096 )
          goto LABEL_23;
        v32 = v29 + j;
        ++v29;
      }
      *v31 = -8192;
      v31[1] = -8192;
      --*(_DWORD *)(v25 + 2008);
      ++*(_DWORD *)(v25 + 2012);
    }
LABEL_23:
    *v15 = a1;
    if ( *(_QWORD *)(a1 - 64) )
    {
      v33 = *(_QWORD *)(a1 - 56);
      **(_QWORD **)(a1 - 48) = v33;
      if ( v33 )
        *(_QWORD *)(v33 + 16) = *(_QWORD *)(a1 - 48);
    }
    *(_QWORD *)(a1 - 64) = v5;
    if ( v5 )
    {
      v34 = *(_QWORD *)(v5 + 16);
      *(_QWORD *)(a1 - 56) = v34;
      if ( v34 )
        *(_QWORD *)(v34 + 16) = a1 - 56;
      *(_QWORD *)(a1 - 48) = v5 + 16;
      *(_QWORD *)(v5 + 16) = a1 - 64;
    }
    if ( *(_QWORD *)(a1 - 32) )
    {
      v35 = *(_QWORD *)(a1 - 24);
      **(_QWORD **)(a1 - 16) = v35;
      if ( v35 )
        *(_QWORD *)(v35 + 16) = *(_QWORD *)(a1 - 16);
    }
    *(_QWORD *)(a1 - 32) = v3;
    if ( v3 )
    {
      v36 = *(_QWORD *)(v3 + 16);
      *(_QWORD *)(a1 - 24) = v36;
      if ( v36 )
        *(_QWORD *)(v36 + 16) = a1 - 24;
      *(_QWORD *)(a1 - 16) = v3 + 16;
      *(_QWORD *)(v3 + 16) = a1 - 32;
      v3 = *(_QWORD *)(a1 - 32);
    }
    *(_WORD *)(v3 + 2) = (*(_WORD *)(v3 + 2) + 1) & 0x7FFF | *(_WORD *)(v3 + 2) & 0x8000;
  }
  return v16;
}
