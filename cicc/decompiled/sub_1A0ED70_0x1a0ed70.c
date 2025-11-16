// Function: sub_1A0ED70
// Address: 0x1a0ed70
//
__int64 __fastcall sub_1A0ED70(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  _QWORD *v5; // rax
  __int64 **v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 i; // r12
  __int64 v14; // rbx
  __int64 v15; // r15
  _QWORD *v16; // rax
  const char *v17; // rbx
  __int64 v18; // rax
  const char **v19; // rbx
  const char **v20; // rbx
  const char **v21; // rdi
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 j; // r12
  const char *v25; // r14
  __int64 v26; // rax
  const char **v27; // rbx
  const char **v28; // rbx
  const char **v29; // rdi
  __int64 v30; // [rsp+10h] [rbp-70h]
  _QWORD *v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+20h] [rbp-60h]
  const char *v33; // [rsp+30h] [rbp-50h] BYREF
  const char **v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h]

  if ( sub_15E4F60(a2) || (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v3 = *(_QWORD *)(a2 + 80);
  if ( !v3 )
    BUG();
  v4 = *(_QWORD *)(v3 + 24);
  v32 = v3 - 24;
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    if ( *(_BYTE *)(v4 - 8) != 53 )
      break;
    v4 = *(_QWORD *)(v4 + 8);
  }
  v5 = (_QWORD *)sub_15E0530(a2);
  v6 = (__int64 **)sub_1643350(v5);
  v9 = sub_15A06D0(v6, a2, v7, v8);
  v10 = (_QWORD *)sub_15E0530(a2);
  v11 = sub_1643350(v10);
  LOWORD(v35) = 259;
  v33 = "reg2mem alloca point";
  v31 = sub_1648A60(56, 1u);
  if ( v31 )
    sub_15FD590((__int64)v31, v9, v11, (__int64)&v33, v4 - 24);
  v35 = 0;
  v34 = &v33;
  v33 = (const char *)&v33;
  v12 = *(_QWORD *)(a2 + 80);
  v30 = a2 + 72;
  if ( v12 != a2 + 72 )
  {
    do
    {
      if ( !v12 )
        BUG();
      for ( i = *(_QWORD *)(v12 + 24); v12 + 16 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        v14 = *(_QWORD *)(i + 16);
        if ( *(_BYTE *)(i - 8) != 53 || v32 != v14 )
        {
          v15 = *(_QWORD *)(i - 16);
          if ( v15 )
          {
            while ( 1 )
            {
              v16 = sub_1648700(v15);
              if ( v14 != v16[5] || *((_BYTE *)v16 + 16) == 77 )
                break;
              v15 = *(_QWORD *)(v15 + 8);
              if ( !v15 )
                goto LABEL_22;
            }
            v17 = v33;
            v18 = sub_22077B0(24);
            *(_QWORD *)(v18 + 16) = i - 24;
            sub_2208C80(v18, v17);
            ++v35;
          }
        }
LABEL_22:
        ;
      }
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v30 != v12 );
    v19 = (const char **)v33;
    if ( v33 != (const char *)&v33 )
    {
      do
      {
        sub_1AC3CB0(v19[2], 0, v31);
        v19 = (const char **)*v19;
      }
      while ( v19 != &v33 );
      v20 = (const char **)v33;
      if ( v33 != (const char *)&v33 )
      {
        do
        {
          v21 = v20;
          v20 = (const char **)*v20;
          j_j___libc_free_0(v21, 24);
        }
        while ( v20 != &v33 );
      }
    }
    v35 = 0;
    v22 = *(_QWORD *)(a2 + 80);
    v34 = &v33;
    v33 = (const char *)&v33;
    if ( v30 != v22 )
    {
      do
      {
        if ( !v22 )
          BUG();
        v23 = *(_QWORD *)(v22 + 24);
        for ( j = v22 + 16; j != v23; v23 = *(_QWORD *)(v23 + 8) )
        {
          while ( 1 )
          {
            if ( !v23 )
              BUG();
            if ( *(_BYTE *)(v23 - 8) == 77 )
              break;
            v23 = *(_QWORD *)(v23 + 8);
            if ( j == v23 )
              goto LABEL_36;
          }
          v25 = v33;
          v26 = sub_22077B0(24);
          *(_QWORD *)(v26 + 16) = v23 - 24;
          sub_2208C80(v26, v25);
          ++v35;
        }
LABEL_36:
        v22 = *(_QWORD *)(v22 + 8);
      }
      while ( v30 != v22 );
      v27 = (const char **)v33;
      if ( v33 != (const char *)&v33 )
      {
        do
        {
          sub_1AC3A00(v27[2], v31);
          v27 = (const char **)*v27;
        }
        while ( v27 != &v33 );
        v28 = (const char **)v33;
        if ( v33 != (const char *)&v33 )
        {
          do
          {
            v29 = v28;
            v28 = (const char **)*v28;
            j_j___libc_free_0(v29, 24);
          }
          while ( v28 != &v33 );
        }
      }
    }
  }
  return 1;
}
