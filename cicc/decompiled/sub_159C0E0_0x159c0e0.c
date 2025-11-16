// Function: sub_159C0E0
// Address: 0x159c0e0
//
__int64 __fastcall sub_159C0E0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  char v4; // al
  __int64 v5; // r13
  __int64 result; // rax
  __int64 v7; // r15
  int v8; // eax
  int v9; // eax
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r13
  unsigned __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 j; // rdx
  __int64 v26; // rdx
  __int64 i; // rdx
  __int64 v28; // [rsp+8h] [rbp-48h]
  _QWORD v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *a1;
  v4 = sub_1598260(*a1 + 136, a2, v29);
  v5 = v29[0];
  if ( v4 )
  {
    result = *(_QWORD *)(v29[0] + 16LL);
    if ( result )
      return result;
    goto LABEL_11;
  }
  v7 = *(unsigned int *)(v3 + 160);
  v8 = *(_DWORD *)(v3 + 152);
  ++*(_QWORD *)(v3 + 136);
  v9 = v8 + 1;
  if ( 4 * v9 >= (unsigned int)(3 * v7) )
  {
    v18 = *(_QWORD *)(v3 + 144);
    v10 = 2 * v7;
    v19 = (((((((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
              | (unsigned int)(v10 - 1)
              | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
            | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
            | (unsigned int)(v10 - 1)
            | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 8)
          | (((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
            | (unsigned int)(v10 - 1)
            | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
          | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
          | (unsigned int)(v10 - 1)
          | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 16)
        | (((((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
            | (unsigned int)(v10 - 1)
            | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
          | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
          | (unsigned int)(v10 - 1)
          | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 8)
        | (((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
          | (unsigned int)(v10 - 1)
          | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
        | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
        | (unsigned int)(v10 - 1)
        | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1);
    v20 = v19 + 1;
    if ( v20 < 0x40 )
      v20 = 64;
    *(_DWORD *)(v3 + 160) = v20;
    v21 = sub_22077B0(24LL * v20);
    *(_QWORD *)(v3 + 144) = v21;
    if ( !v18 )
    {
      v26 = *(unsigned int *)(v3 + 160);
      *(_QWORD *)(v3 + 152) = 0;
      for ( i = v21 + 24 * v26; i != v21; v21 += 24 )
      {
        if ( v21 )
        {
          *(_DWORD *)(v21 + 8) = 0;
          *(_QWORD *)v21 = 0;
        }
      }
      goto LABEL_27;
    }
  }
  else
  {
    if ( (int)v7 - *(_DWORD *)(v3 + 156) - v9 > (unsigned int)v7 >> 3 )
      goto LABEL_6;
    v18 = *(_QWORD *)(v3 + 144);
    v22 = ((((((((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
               | (unsigned int)(v7 - 1)
               | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
             | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
             | (unsigned int)(v7 - 1)
             | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 8)
           | (((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
             | (unsigned int)(v7 - 1)
             | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
           | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
           | (unsigned int)(v7 - 1)
           | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 16)
         | (((((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
             | (unsigned int)(v7 - 1)
             | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
           | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
           | (unsigned int)(v7 - 1)
           | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 8)
         | (((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
           | (unsigned int)(v7 - 1)
           | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
         | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
         | (unsigned int)(v7 - 1)
         | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1))
        + 1;
    if ( (unsigned int)v22 < 0x40 )
      LODWORD(v22) = 64;
    *(_DWORD *)(v3 + 160) = v22;
    v23 = sub_22077B0(24LL * (unsigned int)v22);
    *(_QWORD *)(v3 + 144) = v23;
    if ( !v18 )
    {
      v24 = *(unsigned int *)(v3 + 160);
      *(_QWORD *)(v3 + 152) = 0;
      for ( j = v23 + 24 * v24; j != v23; v23 += 24 )
      {
        if ( v23 )
        {
          *(_DWORD *)(v23 + 8) = 0;
          *(_QWORD *)v23 = 0;
        }
      }
      goto LABEL_27;
    }
  }
  sub_159BF50(v3 + 136, v18, v18 + 24 * v7);
  j___libc_free_0(v18);
LABEL_27:
  sub_1598260(v3 + 136, a2, v29);
  v5 = v29[0];
  v9 = *(_DWORD *)(v3 + 152) + 1;
LABEL_6:
  *(_DWORD *)(v3 + 152) = v9;
  if ( (*(_DWORD *)(v5 + 8) || *(_QWORD *)v5) && (--*(_DWORD *)(v3 + 156), *(_DWORD *)(v5 + 8) > 0x40u)
    || *(_DWORD *)(a2 + 8) > 0x40u )
  {
    sub_16A51C0(v5, a2);
  }
  else
  {
    v14 = *(_QWORD *)a2;
    *(_QWORD *)v5 = *(_QWORD *)a2;
    v15 = *(unsigned int *)(a2 + 8);
    *(_DWORD *)(v5 + 8) = v15;
    v16 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
    if ( (unsigned int)v15 > 0x40 )
    {
      v17 = (unsigned int)((unsigned __int64)(v15 + 63) >> 6) - 1;
      *(_QWORD *)(v14 + 8 * v17) &= v16;
    }
    else
    {
      *(_QWORD *)v5 = v16 & v14;
    }
  }
  *(_QWORD *)(v5 + 16) = 0;
LABEL_11:
  v11 = sub_1644900(a1, *(unsigned int *)(a2 + 8));
  result = sub_1648A60(40, 0);
  if ( result )
  {
    v28 = result;
    sub_1594070(result, v11, a2);
    result = v28;
  }
  v12 = *(_QWORD *)(v5 + 16);
  *(_QWORD *)(v5 + 16) = result;
  if ( v12 )
  {
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
    {
      v13 = *(_QWORD *)(v12 + 24);
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
    sub_164BE60(v12);
    sub_1648B90(v12);
    return *(_QWORD *)(v5 + 16);
  }
  return result;
}
