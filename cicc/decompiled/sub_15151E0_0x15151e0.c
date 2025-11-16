// Function: sub_15151E0
// Address: 0x15151e0
//
void __fastcall sub_15151E0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int i; // r12d
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned int v8; // edx
  unsigned int j; // r15d
  _BYTE *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  const char *v15; // r13
  const char *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // r14
  _BYTE *v20; // rdx
  __int64 v21; // [rsp-70h] [rbp-70h]
  int v22; // [rsp-68h] [rbp-68h]
  __int64 v23; // [rsp-60h] [rbp-60h]
  __int64 v24; // [rsp-60h] [rbp-60h]
  const char *v25; // [rsp-58h] [rbp-58h] BYREF
  __int64 v26; // [rsp-50h] [rbp-50h]
  _BYTE v27[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)(a1 + 1010) )
  {
    v25 = "llvm.dbg.cu";
    v2 = *(_QWORD *)(a1 + 248);
    v27[1] = 1;
    v27[0] = 3;
    v23 = sub_1632310(v2, &v25);
    if ( v23 )
    {
      v22 = sub_161F520(v23, &v25, v3, v4);
      if ( v22 )
      {
        for ( i = 0; i != v22; ++i )
        {
          v6 = sub_161F530(v23, i);
          v7 = *(_QWORD *)(v6 + 8 * (6LL - *(unsigned int *)(v6 + 8)));
          if ( v7 )
          {
            if ( *(_BYTE *)v7 == 4 )
            {
              v8 = *(_DWORD *)(v7 + 8);
              if ( v8 )
              {
                for ( j = 0; j < v8; ++j )
                {
                  v10 = *(_BYTE **)(v7 + 8 * (j - (unsigned __int64)v8));
                  if ( v10 && *v10 == 24 )
                  {
                    v11 = sub_15C4420(*(_QWORD *)(a1 + 240), 0, 0, 0, 1);
                    v12 = sub_15C5570(*(_QWORD *)(a1 + 240), v10, v11, 1, 1);
                    sub_1630830(v7, j, v12);
                    v8 = *(_DWORD *)(v7 + 8);
                  }
                }
              }
            }
          }
        }
      }
    }
    v13 = *(_QWORD *)(a1 + 248);
    v21 = v13 + 8;
    v24 = *(_QWORD *)(v13 + 16);
    if ( v13 + 8 != v24 )
    {
      do
      {
        v14 = v24 - 56;
        if ( !v24 )
          v14 = 0;
        v25 = v27;
        v26 = 0x100000000LL;
        sub_1626560(v14, 0, &v25);
        sub_1626EF0(v14, 0);
        v15 = &v25[8 * (unsigned int)v26];
        if ( v25 != v15 )
        {
          v16 = v25;
          do
          {
            while ( 1 )
            {
              v19 = *(_BYTE **)v16;
              if ( !*(_QWORD *)v16 || *v19 != 24 )
                break;
              v16 += 8;
              v17 = sub_15C4420(*(_QWORD *)(a1 + 240), 0, 0, 0, 1);
              v18 = sub_15C5570(*(_QWORD *)(a1 + 240), v19, v17, 1, 1);
              sub_16267C0(v14, 0, v18);
              if ( v15 == v16 )
                goto LABEL_23;
            }
            v20 = *(_BYTE **)v16;
            v16 += 8;
            sub_16267C0(v14, 0, v20);
          }
          while ( v15 != v16 );
LABEL_23:
          v15 = v25;
        }
        if ( v15 != v27 )
          _libc_free((unsigned __int64)v15);
        v24 = *(_QWORD *)(v24 + 8);
      }
      while ( v21 != v24 );
    }
  }
}
