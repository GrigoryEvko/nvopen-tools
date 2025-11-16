// Function: sub_BF6140
// Address: 0xbf6140
//
unsigned __int64 __fastcall sub_BF6140(__int64 a1, _BYTE *a2, const char *a3, char a4)
{
  const char *v5; // r12
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // esi
  const char **v11; // rdx
  const char *v12; // r9
  __int64 *v14; // r14
  __int64 v15; // r15
  _BYTE *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned int v19; // esi
  unsigned __int64 v20; // r13
  __int64 v21; // rcx
  unsigned int v22; // edx
  const char **v23; // rax
  const char *v24; // r8
  int v25; // edx
  int v26; // r10d
  int v27; // r11d
  const char **v28; // r9
  int v29; // eax
  int v30; // edx
  const char **v31; // [rsp+8h] [rbp-68h] BYREF
  const char *v32; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v33; // [rsp+18h] [rbp-58h]
  char v34; // [rsp+30h] [rbp-40h]
  char v35; // [rsp+31h] [rbp-3Fh]

  v5 = a3;
  if ( (*(a3 - 16) & 2) != 0 )
    v7 = *((_DWORD *)a3 - 6);
  else
    v7 = (*((_WORD *)a3 - 8) >> 6) & 0xF;
  if ( v7 <= 1 )
  {
    v14 = *(__int64 **)a1;
    if ( *(_QWORD *)a1 )
    {
      v35 = 1;
      v32 = "Base nodes must have at least two operands";
      v34 = 3;
      v15 = *v14;
      if ( !*v14 )
      {
        *((_BYTE *)v14 + 152) = 1;
        return 0xFFFFFFFF00000001LL;
      }
      sub_CA0E80(&v32, *v14);
      v16 = *(_BYTE **)(v15 + 32);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
      {
        sub_CB5D20(v15, 10);
      }
      else
      {
        *(_QWORD *)(v15 + 32) = v16 + 1;
        *v16 = 10;
      }
      v17 = *v14;
      *((_BYTE *)v14 + 152) = 1;
      if ( v17 )
      {
        sub_BDBD80((__int64)v14, a2);
        sub_BD9900(v14, v5);
      }
    }
    return 0xFFFFFFFF00000001LL;
  }
  v8 = *(unsigned int *)(a1 + 32);
  v9 = *(_QWORD *)(a1 + 16);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (const char **)(v9 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == v5 )
    {
LABEL_6:
      if ( v11 != (const char **)(v9 + 16 * v8) )
        return (unsigned __int64)v11[1];
    }
    else
    {
      v25 = 1;
      while ( v12 != (const char *)-4096LL )
      {
        v26 = v25 + 1;
        v10 = (v8 - 1) & (v25 + v10);
        v11 = (const char **)(v9 + 16LL * v10);
        v12 = *v11;
        if ( *v11 == v5 )
          goto LABEL_6;
        v25 = v26;
      }
    }
  }
  v18 = sub_BF5780((__int64 **)a1, a2, v5, a4);
  v19 = *(_DWORD *)(a1 + 32);
  v32 = v5;
  v33 = v18;
  v20 = v18;
  if ( !v19 )
  {
    ++*(_QWORD *)(a1 + 8);
    v31 = 0;
    goto LABEL_35;
  }
  v21 = *(_QWORD *)(a1 + 16);
  v22 = (v19 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v23 = (const char **)(v21 + 16LL * v22);
  v24 = *v23;
  if ( *v23 != v5 )
  {
    v27 = 1;
    v28 = 0;
    while ( v24 != (const char *)-4096LL )
    {
      if ( v24 == (const char *)-8192LL && !v28 )
        v28 = v23;
      v22 = (v19 - 1) & (v27 + v22);
      v23 = (const char **)(v21 + 16LL * v22);
      v24 = *v23;
      if ( *v23 == v5 )
        return v20;
      ++v27;
    }
    if ( !v28 )
      v28 = v23;
    v29 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v30 = v29 + 1;
    v31 = v28;
    if ( 4 * (v29 + 1) < 3 * v19 )
    {
      if ( v19 - *(_DWORD *)(a1 + 28) - v30 > v19 >> 3 )
      {
LABEL_31:
        *(_DWORD *)(a1 + 24) = v30;
        if ( *v28 != (const char *)-4096LL )
          --*(_DWORD *)(a1 + 28);
        *v28 = v5;
        v28[1] = (const char *)v33;
        return v20;
      }
LABEL_36:
      sub_BF51A0(a1 + 8, v19);
      sub_BF04F0(a1 + 8, (__int64 *)&v32, &v31);
      v5 = v32;
      v28 = v31;
      v30 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_31;
    }
LABEL_35:
    v19 *= 2;
    goto LABEL_36;
  }
  return v20;
}
