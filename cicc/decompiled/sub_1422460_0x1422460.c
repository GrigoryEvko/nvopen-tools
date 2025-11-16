// Function: sub_1422460
// Address: 0x1422460
//
unsigned __int64 *__fastcall sub_1422460(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // r12
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 *v12; // r13
  unsigned __int64 *v13; // rcx
  unsigned __int64 v14; // rdx
  unsigned __int64 *v15; // r15
  unsigned __int64 *result; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned int v19; // edx
  __int64 *v20; // r13
  unsigned __int64 *v21; // rcx
  unsigned __int64 v22; // rdx
  _QWORD *v23; // rdi
  unsigned __int64 *v24; // r12
  unsigned __int64 *v25; // rdi
  unsigned __int64 v26; // rdx
  unsigned __int64 *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx

  v6 = (_QWORD *)a2;
  v8 = *(_QWORD *)(a2 + 64);
  if ( *(_BYTE *)(a2 + 16) == 21 )
    goto LABEL_2;
  v17 = *(unsigned int *)(a1 + 112);
  v18 = *(_QWORD *)(a1 + 96);
  if ( (_DWORD)v17 )
  {
    a5 = 1;
    v19 = (v17 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v20 = (__int64 *)(v18 + 16LL * v19);
    a2 = *v20;
    if ( v8 == *v20 )
      goto LABEL_14;
    while ( a2 != -8 )
    {
      v19 = (v17 - 1) & (a5 + v19);
      v20 = (__int64 *)(v18 + 16LL * v19);
      a2 = *v20;
      if ( v8 == *v20 )
        goto LABEL_14;
      a5 = (unsigned int)(a5 + 1);
    }
  }
  v20 = (__int64 *)(v18 + 16 * v17);
LABEL_14:
  v21 = (unsigned __int64 *)v6[7];
  v22 = v6[6] & 0xFFFFFFFFFFFFFFF8LL;
  *v21 = v22 | *v21 & 7;
  *(_QWORD *)(v22 + 8) = v21;
  v6[7] = 0;
  v6[6] &= 7uLL;
  v23 = (_QWORD *)v20[1];
  if ( v23 == (_QWORD *)(*v23 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    a2 = 16;
    j_j___libc_free_0(v23, 16);
    *v20 = -16;
    --*(_DWORD *)(a1 + 104);
    ++*(_DWORD *)(a1 + 108);
  }
LABEL_2:
  v9 = *(unsigned int *)(a1 + 80);
  v10 = *(_QWORD *)(a1 + 64);
  if ( (_DWORD)v9 )
  {
    a5 = 1;
    v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = (__int64 *)(v10 + 16LL * v11);
    a2 = *v12;
    if ( v8 == *v12 )
      goto LABEL_4;
    while ( a2 != -8 )
    {
      v11 = (v9 - 1) & (a5 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      a2 = *v12;
      if ( v8 == *v12 )
        goto LABEL_4;
      a5 = (unsigned int)(a5 + 1);
    }
  }
  v12 = (__int64 *)(v10 + 16 * v9);
LABEL_4:
  v13 = (unsigned __int64 *)v6[5];
  v14 = v6[4] & 0xFFFFFFFFFFFFFFF8LL;
  *v13 = v14 | *v13 & 7;
  *(_QWORD *)(v14 + 8) = v13;
  v6[4] &= 7uLL;
  v6[5] = 0;
  if ( a3 )
    sub_164BEC0(v6, a2, v14, v13, a5);
  v15 = (unsigned __int64 *)v12[1];
  result = (unsigned __int64 *)(*v15 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v15 == result )
  {
    v24 = (unsigned __int64 *)v15[1];
    while ( v24 != v15 )
    {
      v25 = v24;
      v24 = (unsigned __int64 *)v24[1];
      v26 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
      *v24 = v26 | *v24 & 7;
      *(_QWORD *)(v26 + 8) = v24;
      *v25 &= 7u;
      v25 -= 4;
      v25[5] = 0;
      sub_164BEC0(v25, a2, v26, v13, a5);
    }
    j_j___libc_free_0(v15, 16);
    *v12 = -16;
    result = *(unsigned __int64 **)(a1 + 136);
    --*(_DWORD *)(a1 + 72);
    ++*(_DWORD *)(a1 + 76);
    if ( *(unsigned __int64 **)(a1 + 144) == result )
    {
      v27 = &result[*(unsigned int *)(a1 + 156)];
      if ( result == v27 )
      {
LABEL_36:
        result = v27;
      }
      else
      {
        while ( v8 != *result )
        {
          if ( v27 == ++result )
            goto LABEL_36;
        }
      }
    }
    else
    {
      result = (unsigned __int64 *)sub_16CC9F0(a1 + 128, v8);
      if ( v8 == *result )
      {
        v28 = *(_QWORD *)(a1 + 144);
        if ( v28 == *(_QWORD *)(a1 + 136) )
          v29 = *(unsigned int *)(a1 + 156);
        else
          v29 = *(unsigned int *)(a1 + 152);
        v27 = (unsigned __int64 *)(v28 + 8 * v29);
      }
      else
      {
        result = *(unsigned __int64 **)(a1 + 144);
        if ( result != *(unsigned __int64 **)(a1 + 136) )
          return result;
        result += *(unsigned int *)(a1 + 156);
        v27 = result;
      }
    }
    if ( v27 != result )
    {
      *result = -2;
      ++*(_DWORD *)(a1 + 160);
    }
  }
  return result;
}
