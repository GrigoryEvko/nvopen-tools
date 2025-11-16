// Function: sub_35DE620
// Address: 0x35de620
//
void __fastcall sub_35DE620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r10
  __int64 v7; // r14
  bool v8; // cf
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned int v11; // r15d
  __int64 v12; // rdx
  char v13; // al
  __int64 *v14; // r13
  __int64 *i; // r15
  __int64 v16; // rdi
  __int64 v17; // rdi
  _QWORD *v18; // rax
  char v20; // [rsp+27h] [rbp-69h]
  __int64 v21; // [rsp+28h] [rbp-68h]
  __int64 v22; // [rsp+28h] [rbp-68h]
  __int64 *v23; // [rsp+30h] [rbp-60h] BYREF
  __int64 v24; // [rsp+38h] [rbp-58h]
  _BYTE v25[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(_BYTE *)a3 < 0x1Du;
  v23 = (__int64 *)v25;
  if ( !v8 )
    v6 = a3;
  v24 = 0x400000000LL;
  if ( v7 )
  {
    v20 = 1;
    v9 = a3;
    v10 = v6;
    v11 = 0;
    do
    {
      while ( 1 )
      {
        a6 = *(_QWORD *)(v7 + 24);
        if ( !v10 )
          break;
        v21 = *(_QWORD *)(v7 + 24);
        v13 = sub_B46220(v21, v10);
        a6 = v21;
        if ( !v13 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        v20 = 0;
        if ( !v7 )
          goto LABEL_11;
      }
      v12 = v11;
      if ( (unsigned __int64)v11 + 1 > HIDWORD(v24) )
      {
        v22 = a6;
        sub_C8D5F0((__int64)&v23, v25, v11 + 1LL, 8u, a5, a6);
        v12 = (unsigned int)v24;
        a6 = v22;
      }
      v23[v12] = a6;
      v11 = v24 + 1;
      LODWORD(v24) = v24 + 1;
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v7 );
LABEL_11:
    a4 = (__int64)v23;
    a3 = v11;
    v14 = &v23[v11];
    for ( i = v23; v14 != i; ++i )
    {
      v16 = *i;
      sub_BD2ED0(v16, a2, v9);
    }
    if ( !v20 || *(_BYTE *)a2 <= 0x1Cu )
      goto LABEL_14;
  }
  else if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    return;
  }
  v17 = *(_QWORD *)(a1 + 48);
  if ( !*(_BYTE *)(v17 + 28) )
    goto LABEL_24;
  v18 = *(_QWORD **)(v17 + 8);
  a4 = *(unsigned int *)(v17 + 20);
  a3 = (__int64)&v18[a4];
  if ( v18 != (_QWORD *)a3 )
  {
    while ( a2 != *v18 )
    {
      if ( (_QWORD *)a3 == ++v18 )
        goto LABEL_22;
    }
    goto LABEL_14;
  }
LABEL_22:
  if ( (unsigned int)a4 < *(_DWORD *)(v17 + 16) )
  {
    *(_DWORD *)(v17 + 20) = a4 + 1;
    *(_QWORD *)a3 = a2;
    ++*(_QWORD *)v17;
  }
  else
  {
LABEL_24:
    sub_C8CC70(v17, a2, a3, a4, a5, a6);
  }
LABEL_14:
  if ( v23 != (__int64 *)v25 )
    _libc_free((unsigned __int64)v23);
}
