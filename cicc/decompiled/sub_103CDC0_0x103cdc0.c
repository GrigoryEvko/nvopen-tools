// Function: sub_103CDC0
// Address: 0x103cdc0
//
unsigned __int64 *__fastcall sub_103CDC0(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v4; // r12
  __int64 v6; // r14
  unsigned int v7; // edx
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 *v10; // r13
  unsigned __int64 *v11; // rcx
  unsigned __int64 v12; // rdx
  unsigned __int64 *v13; // r15
  unsigned __int64 *result; // rax
  unsigned int v15; // edx
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 *v18; // r13
  unsigned __int64 *v19; // rcx
  unsigned __int64 v20; // rdx
  _QWORD *v21; // rdi
  int v22; // r8d
  unsigned __int64 *v23; // r12
  unsigned __int64 *v24; // rdi
  unsigned __int64 v25; // rdx
  unsigned __int64 *v26; // rsi
  unsigned __int64 *v27; // rdx
  __int64 v28; // rcx
  int v29; // r8d

  v4 = (_QWORD *)a2;
  v6 = *(_QWORD *)(a2 + 64);
  if ( *(_BYTE *)a2 == 26 )
    goto LABEL_2;
  v15 = *(_DWORD *)(a1 + 120);
  v16 = *(_QWORD *)(a1 + 104);
  if ( v15 )
  {
    v17 = (v15 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v18 = (__int64 *)(v16 + 16LL * v17);
    a2 = *v18;
    if ( *v18 == v6 )
      goto LABEL_10;
    v29 = 1;
    while ( a2 != -4096 )
    {
      v17 = (v15 - 1) & (v29 + v17);
      v18 = (__int64 *)(v16 + 16LL * v17);
      a2 = *v18;
      if ( v6 == *v18 )
        goto LABEL_10;
      ++v29;
    }
  }
  v18 = (__int64 *)(v16 + 16LL * v15);
LABEL_10:
  v19 = (unsigned __int64 *)v4[7];
  v20 = v4[6] & 0xFFFFFFFFFFFFFFF8LL;
  *v19 = v20 | *v19 & 7;
  *(_QWORD *)(v20 + 8) = v19;
  v4[7] = 0;
  v4[6] &= 7uLL;
  v21 = (_QWORD *)v18[1];
  if ( v21 == (_QWORD *)(*v21 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    a2 = 16;
    j_j___libc_free_0(v21, 16);
    *v18 = -8192;
    --*(_DWORD *)(a1 + 112);
    ++*(_DWORD *)(a1 + 116);
  }
LABEL_2:
  v7 = *(_DWORD *)(a1 + 88);
  v8 = *(_QWORD *)(a1 + 72);
  if ( v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    a2 = *v10;
    if ( v6 == *v10 )
      goto LABEL_4;
    v22 = 1;
    while ( a2 != -4096 )
    {
      v9 = (v7 - 1) & (v22 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      a2 = *v10;
      if ( v6 == *v10 )
        goto LABEL_4;
      ++v22;
    }
  }
  v10 = (__int64 *)(v8 + 16LL * v7);
LABEL_4:
  v11 = (unsigned __int64 *)v4[5];
  v12 = v4[4] & 0xFFFFFFFFFFFFFFF8LL;
  *v11 = v12 | *v11 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  v4[4] &= 7uLL;
  v4[5] = 0;
  if ( a3 )
    sub_BD72D0((__int64)v4, a2);
  v13 = (unsigned __int64 *)v10[1];
  result = (unsigned __int64 *)(*v13 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v13 == result )
  {
    v23 = (unsigned __int64 *)v13[1];
    while ( v13 != v23 )
    {
      v24 = v23;
      v23 = (unsigned __int64 *)v23[1];
      v25 = *v24 & 0xFFFFFFFFFFFFFFF8LL;
      *v23 = v25 | *v23 & 7;
      *(_QWORD *)(v25 + 8) = v23;
      *v24 &= 7u;
      v24 -= 4;
      v24[5] = 0;
      sub_BD72D0((__int64)v24, a2);
    }
    j_j___libc_free_0(v13, 16);
    *v10 = -8192;
    --*(_DWORD *)(a1 + 80);
    ++*(_DWORD *)(a1 + 84);
    if ( *(_BYTE *)(a1 + 164) )
    {
      v26 = *(unsigned __int64 **)(a1 + 144);
      v27 = &v26[*(unsigned int *)(a1 + 156)];
      result = v26;
      if ( v26 != v27 )
      {
        while ( v6 != *result )
        {
          if ( v27 == ++result )
            return result;
        }
        v28 = (unsigned int)(*(_DWORD *)(a1 + 156) - 1);
        *(_DWORD *)(a1 + 156) = v28;
        *result = v26[v28];
        ++*(_QWORD *)(a1 + 136);
      }
    }
    else
    {
      result = (unsigned __int64 *)sub_C8CA60(a1 + 136, v6);
      if ( result )
      {
        *result = -2;
        ++*(_DWORD *)(a1 + 160);
        ++*(_QWORD *)(a1 + 136);
      }
    }
  }
  return result;
}
