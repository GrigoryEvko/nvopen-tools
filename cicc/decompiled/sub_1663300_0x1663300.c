// Function: sub_1663300
// Address: 0x1663300
//
unsigned __int64 __fastcall sub_1663300(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned __int8 *v5; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // esi
  unsigned __int8 **v10; // rdx
  unsigned __int8 *v11; // r9
  __int64 *v13; // r14
  __int64 v14; // r15
  _BYTE *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned int v18; // esi
  unsigned __int64 v19; // r13
  __int64 v20; // rcx
  unsigned int v21; // edx
  unsigned __int8 **v22; // rax
  unsigned __int8 *v23; // r8
  int v24; // edx
  int v25; // r10d
  int v26; // r11d
  unsigned __int8 **v27; // r9
  int v28; // eax
  int v29; // edx
  unsigned __int8 **v30; // [rsp+8h] [rbp-58h] BYREF
  const char *v31; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v32; // [rsp+18h] [rbp-48h]
  char v33; // [rsp+20h] [rbp-40h]
  char v34; // [rsp+21h] [rbp-3Fh]

  v5 = (unsigned __int8 *)a3;
  if ( *(_DWORD *)(a3 + 8) > 1u )
  {
    v7 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD *)(a1 + 16);
      v9 = (v7 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v10 = (unsigned __int8 **)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == v5 )
      {
LABEL_4:
        if ( v10 != (unsigned __int8 **)(v8 + 16 * v7) )
          return (unsigned __int64)v10[1];
      }
      else
      {
        v24 = 1;
        while ( v11 != (unsigned __int8 *)-8LL )
        {
          v25 = v24 + 1;
          v9 = (v7 - 1) & (v24 + v9);
          v10 = (unsigned __int8 **)(v8 + 16LL * v9);
          v11 = *v10;
          if ( *v10 == v5 )
            goto LABEL_4;
          v24 = v25;
        }
      }
    }
    v17 = sub_1662A80((__int64 **)a1, a2, (__int64)v5, a4);
    v18 = *(_DWORD *)(a1 + 32);
    v31 = (const char *)v5;
    v32 = v17;
    v19 = v17;
    if ( v18 )
    {
      v20 = *(_QWORD *)(a1 + 16);
      v21 = (v18 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v22 = (unsigned __int8 **)(v20 + 16LL * v21);
      v23 = *v22;
      if ( *v22 == v5 )
        return v19;
      v26 = 1;
      v27 = 0;
      while ( v23 != (unsigned __int8 *)-8LL )
      {
        if ( v23 == (unsigned __int8 *)-16LL && !v27 )
          v27 = v22;
        v21 = (v18 - 1) & (v26 + v21);
        v22 = (unsigned __int8 **)(v20 + 16LL * v21);
        v23 = *v22;
        if ( *v22 == v5 )
          return v19;
        ++v26;
      }
      if ( !v27 )
        v27 = v22;
      v28 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v29 = v28 + 1;
      if ( 4 * (v28 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a1 + 28) - v29 > v18 >> 3 )
        {
LABEL_28:
          *(_DWORD *)(a1 + 24) = v29;
          if ( *v27 != (unsigned __int8 *)-8LL )
            --*(_DWORD *)(a1 + 28);
          *v27 = v5;
          v27[1] = (unsigned __int8 *)v32;
          return v19;
        }
LABEL_33:
        sub_16624F0(a1 + 8, v18);
        sub_165C450(a1 + 8, (__int64 *)&v31, &v30);
        v27 = v30;
        v5 = (unsigned __int8 *)v31;
        v29 = *(_DWORD *)(a1 + 24) + 1;
        goto LABEL_28;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    v18 *= 2;
    goto LABEL_33;
  }
  v13 = *(__int64 **)a1;
  if ( !*(_QWORD *)a1 )
    return 0xFFFFFFFF00000001LL;
  v34 = 1;
  v31 = "Base nodes must have at least two operands";
  v33 = 3;
  v14 = *v13;
  if ( *v13 )
  {
    sub_16E2CE0(&v31, *v13);
    v15 = *(_BYTE **)(v14 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
    {
      sub_16E7DE0(v14, 10);
    }
    else
    {
      *(_QWORD *)(v14 + 24) = v15 + 1;
      *v15 = 10;
    }
    v16 = *v13;
    *((_BYTE *)v13 + 72) = 1;
    if ( v16 )
    {
      sub_164FA80(v13, a2);
      sub_164ED40(v13, v5);
    }
    return 0xFFFFFFFF00000001LL;
  }
  *((_BYTE *)v13 + 72) = 1;
  return 0xFFFFFFFF00000001LL;
}
