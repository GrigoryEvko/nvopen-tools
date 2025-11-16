// Function: sub_159CB40
// Address: 0x159cb40
//
__int64 __fastcall sub_159CB40(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // [rsp+8h] [rbp-48h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = 40LL * (unsigned int)v5;
  *(_QWORD *)(a1 + 8) = sub_22077B0(v6);
  if ( v4 )
  {
    sub_159C590(a1, v4, v4 + 40 * v3);
    return j___libc_free_0(v4);
  }
  *(_QWORD *)(a1 + 16) = 0;
  v8 = sub_16982B0(v6, a2);
  v11 = sub_16982C0(v6, a2, v9, v10);
  v12 = v11;
  if ( v8 == v11 )
    sub_169C630(&v19, v11, 1);
  else
    sub_1699170(&v19, v8, 1);
  v13 = *(_QWORD *)(a1 + 8);
  result = 5LL * *(unsigned int *)(a1 + 24);
  v14 = v13 + 40LL * *(unsigned int *)(a1 + 24);
  if ( v13 != v14 )
  {
    while ( 1 )
    {
      if ( !v13 )
        goto LABEL_10;
      v15 = v13 + 8;
      if ( v12 == v19 )
      {
        result = sub_169C6E0(v15, &v19);
        v13 += 40;
        if ( v14 == v13 )
          break;
      }
      else
      {
        result = sub_16986C0(v15, &v19);
LABEL_10:
        v13 += 40;
        if ( v14 == v13 )
          break;
      }
    }
  }
  if ( v12 != v19 )
    return sub_1698460(&v19);
  v16 = v20;
  if ( v20 )
  {
    v17 = 32LL * *(_QWORD *)(v20 - 8);
    v18 = v20 + v17;
    if ( v20 != v20 + v17 )
    {
      do
      {
        v18 -= 32;
        sub_127D120((_QWORD *)(v18 + 8));
      }
      while ( v16 != v18 );
    }
    return j_j_j___libc_free_0_0(v16 - 8);
  }
  return result;
}
