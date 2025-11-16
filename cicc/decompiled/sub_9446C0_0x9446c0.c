// Function: sub_9446C0
// Address: 0x9446c0
//
__int64 __fastcall sub_9446C0(__int64 a1, __int64 a2, const char *a3, __int64 a4, int a5, char a6)
{
  unsigned __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int64 v13; // r15
  unsigned int v14; // esi
  __int64 v15; // rdi
  int v16; // r11d
  __int64 *v17; // r9
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  __int64 v20; // rdx
  unsigned __int64 *v21; // rax
  __int64 result; // rax
  size_t v23; // rax
  __int64 v24; // rcx
  size_t v25; // r8
  _QWORD *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  char v29; // r8
  __int64 v30; // rcx
  int v31; // eax
  int v32; // edx
  unsigned int v33; // eax
  unsigned int v34; // eax
  __int64 v35; // rax
  _QWORD *v36; // rdi
  int v37; // eax
  int v38; // esi
  __int64 v39; // rdi
  unsigned int v40; // ecx
  __int64 v41; // rax
  int v42; // r10d
  __int64 *v43; // r8
  int v44; // eax
  int v45; // ecx
  __int64 v46; // rsi
  int v47; // r8d
  unsigned int v48; // r13d
  __int64 *v49; // rdi
  __int64 v50; // rax
  char n; // [rsp+10h] [rbp-A0h]
  size_t na; // [rsp+10h] [rbp-A0h]
  _QWORD *v55; // [rsp+30h] [rbp-80h] BYREF
  size_t v56; // [rsp+38h] [rbp-78h]
  _QWORD v57[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v58[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v59; // [rsp+70h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 120);
  if ( sub_91B770(v10) || a6 )
  {
    v13 = a4;
    goto LABEL_4;
  }
  sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, v10, v11, v12);
  v55 = v57;
  if ( !a3 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v23 = strlen(a3);
  v58[0] = v23;
  v25 = v23;
  if ( v23 > 0xF )
  {
    na = v23;
    v35 = sub_22409D0(&v55, v58, 0);
    v25 = na;
    v55 = (_QWORD *)v35;
    v36 = (_QWORD *)v35;
    v57[0] = v58[0];
    goto LABEL_49;
  }
  if ( v23 != 1 )
  {
    if ( !v23 )
    {
      v26 = v57;
      goto LABEL_17;
    }
    v36 = v57;
LABEL_49:
    memcpy(v36, a3, v25);
    v23 = v58[0];
    v26 = v55;
    goto LABEL_17;
  }
  LOBYTE(v57[0]) = *a3;
  v26 = v57;
LABEL_17:
  v56 = v23;
  *((_BYTE *)v26 + v23) = 0;
  if ( 0x3FFFFFFFFFFFFFFFLL - v56 <= 4 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v55, ".addr", 5, v24);
  v58[0] = "tmp";
  v59 = 259;
  v28 = sub_921D70(a1, v10, (__int64)v58, v27);
  v13 = v28;
  v59 = 257;
  if ( *(_BYTE *)v55 )
  {
    v58[0] = v55;
    LOBYTE(v59) = 3;
  }
  sub_BD6B50(v28, v58);
  if ( (*(_BYTE *)(a2 + 88) & 4) != 0 )
    goto LABEL_42;
  if ( (*(_BYTE *)(v10 + 140) & 0xFB) != 8 )
    goto LABEL_22;
  if ( (sub_8D4C10(v10, dword_4F077C4 != 2) & 2) == 0 || *(char *)(a2 + 169) >= 0 )
  {
LABEL_42:
    if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
    {
      v33 = (unsigned int)sub_8D4C10(v10, dword_4F077C4 != 2) >> 1;
      v29 = v33 & 1;
      if ( *(char *)(v10 + 142) >= 0 && *(_BYTE *)(v10 + 140) == 12 )
      {
        n = v33 & 1;
        v34 = sub_8D4AB0(v10);
        v29 = n;
        v30 = v34;
        goto LABEL_24;
      }
LABEL_23:
      v30 = *(unsigned int *)(v10 + 136);
LABEL_24:
      sub_923130(a1, a4, v13, v30, v29);
      goto LABEL_25;
    }
LABEL_22:
    v29 = 0;
    goto LABEL_23;
  }
LABEL_25:
  if ( v55 != v57 )
    j_j___libc_free_0(v55, v57[0] + 1LL);
LABEL_4:
  v59 = 257;
  if ( *a3 )
  {
    v58[0] = a3;
    LOBYTE(v59) = 3;
  }
  sub_BD6B50(a4, v58);
  if ( sub_9439D0(a1, a2) )
    sub_91B8A0("unexpected: declaration for variable already exists!", (_DWORD *)(a2 + 64), 1);
  v14 = *(_DWORD *)(a1 + 24);
  if ( !v14 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_51;
  }
  v15 = *(_QWORD *)(a1 + 8);
  v16 = 1;
  v17 = 0;
  v18 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (_QWORD *)(v15 + 16LL * v18);
  v20 = *v19;
  if ( a2 == *v19 )
  {
LABEL_10:
    v21 = v19 + 1;
    goto LABEL_11;
  }
  while ( v20 != -4096 )
  {
    if ( !v17 && v20 == -8192 )
      v17 = v19;
    v18 = (v14 - 1) & (v16 + v18);
    v19 = (_QWORD *)(v15 + 16LL * v18);
    v20 = *v19;
    if ( a2 == *v19 )
      goto LABEL_10;
    ++v16;
  }
  if ( !v17 )
    v17 = v19;
  v31 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v14 )
  {
LABEL_51:
    sub_9437F0(a1, 2 * v14);
    v37 = *(_DWORD *)(a1 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 8);
      v40 = (v37 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = *(_DWORD *)(a1 + 16) + 1;
      v17 = (__int64 *)(v39 + 16LL * v40);
      v41 = *v17;
      if ( a2 != *v17 )
      {
        v42 = 1;
        v43 = 0;
        while ( v41 != -4096 )
        {
          if ( !v43 && v41 == -8192 )
            v43 = v17;
          v40 = v38 & (v42 + v40);
          v17 = (__int64 *)(v39 + 16LL * v40);
          v41 = *v17;
          if ( a2 == *v17 )
            goto LABEL_37;
          ++v42;
        }
        if ( v43 )
          v17 = v43;
      }
      goto LABEL_37;
    }
    goto LABEL_77;
  }
  if ( v14 - *(_DWORD *)(a1 + 20) - v32 <= v14 >> 3 )
  {
    sub_9437F0(a1, v14);
    v44 = *(_DWORD *)(a1 + 24);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 8);
      v47 = 1;
      v48 = (v44 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v49 = 0;
      v32 = *(_DWORD *)(a1 + 16) + 1;
      v17 = (__int64 *)(v46 + 16LL * v48);
      v50 = *v17;
      if ( a2 != *v17 )
      {
        while ( v50 != -4096 )
        {
          if ( !v49 && v50 == -8192 )
            v49 = v17;
          v48 = v45 & (v47 + v48);
          v17 = (__int64 *)(v46 + 16LL * v48);
          v50 = *v17;
          if ( a2 == *v17 )
            goto LABEL_37;
          ++v47;
        }
        if ( v49 )
          v17 = v49;
      }
      goto LABEL_37;
    }
LABEL_77:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_37:
  *(_DWORD *)(a1 + 16) = v32;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v17 = a2;
  v21 = (unsigned __int64 *)(v17 + 1);
  v17[1] = 0;
LABEL_11:
  *v21 = v13;
  result = dword_4D046B4;
  if ( dword_4D046B4 )
    return sub_9433F0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 368LL), a2, v13, a5, a1 + 48);
  return result;
}
