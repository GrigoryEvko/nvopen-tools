// Function: sub_15D5DA0
// Address: 0x15d5da0
//
__int64 __fastcall sub_15D5DA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  __int64 v5; // rbx
  __int64 *v6; // r15
  unsigned int v7; // edx
  __int64 v8; // rsi
  __int64 v9; // rax
  _QWORD *v10; // r14
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // rbx
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  unsigned __int64 v16; // rdi
  __int64 result; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  unsigned __int64 v20; // rdi
  char *v22[2]; // [rsp+20h] [rbp-A0h] BYREF
  char v23; // [rsp+30h] [rbp-90h] BYREF
  __int64 v24[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v25; // [rsp+60h] [rbp-60h]
  __int64 v26; // [rsp+68h] [rbp-58h]
  _QWORD *v27; // [rsp+70h] [rbp-50h]
  __int64 v28; // [rsp+78h] [rbp-48h]
  unsigned int v29; // [rsp+80h] [rbp-40h]
  __int64 v30; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 88);
  sub_15CE0F0(a1 + 48);
  *(_QWORD *)(a1 + 88) = v3;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 100) = 0;
  v4 = (_QWORD *)sub_22077B0(8);
  v26 = 0;
  *v4 = 0;
  v24[0] = (__int64)v4;
  v25 = v4 + 1;
  v24[1] = (__int64)(v4 + 1);
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  sub_15D57B0(v22, a1, 0);
  sub_15CBF70(a1, v22);
  if ( v22[0] != &v23 )
    _libc_free((unsigned __int64)v22[0]);
  sub_15D45B0((__int64)v24);
  v5 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v5 )
  {
    v6 = *(__int64 **)a1;
    v7 = 1;
    do
    {
      v8 = *v6++;
      v7 = sub_15D5190((__int64)v24, v8, v7, (unsigned __int8 (__fastcall *)(char *))sub_15CBC50, 0);
    }
    while ( (__int64 *)v5 != v6 );
  }
  sub_15D4AE0(v24, a1, 0);
  if ( a2 )
    *(_BYTE *)(a2 + 144) = 1;
  if ( *(_DWORD *)(a1 + 8) )
  {
    v22[0] = 0;
    v9 = sub_22077B0(56);
    v10 = (_QWORD *)v9;
    if ( v9 )
    {
      *(_QWORD *)v9 = 0;
      *(_QWORD *)(v9 + 8) = 0;
      *(_DWORD *)(v9 + 16) = 0;
      *(_QWORD *)(v9 + 24) = 0;
      *(_QWORD *)(v9 + 32) = 0;
      *(_QWORD *)(v9 + 40) = 0;
      *(_QWORD *)(v9 + 48) = -1;
    }
    v11 = sub_15CFF10(a1 + 48, (__int64 *)v22);
    v12 = v11[1];
    v13 = v11;
    v11[1] = (__int64)v10;
    if ( v12 )
    {
      sub_15CBC60(v12);
      v10 = (_QWORD *)v13[1];
    }
    *(_QWORD *)(a1 + 80) = v10;
    sub_15D4EE0((__int64)v24, a1, v10);
    if ( v29 )
    {
      v14 = v27;
      v15 = &v27[9 * v29];
      do
      {
        if ( *v14 != -8 && *v14 != -16 )
        {
          v16 = v14[5];
          if ( (_QWORD *)v16 != v14 + 7 )
            _libc_free(v16);
        }
        v14 += 9;
      }
      while ( v15 != v14 );
    }
    result = j___libc_free_0(v27);
    if ( v24[0] )
      return j_j___libc_free_0(v24[0], (char *)v25 - v24[0]);
  }
  else
  {
    if ( v29 )
    {
      v18 = v27;
      v19 = &v27[9 * v29];
      do
      {
        if ( *v18 != -16 && *v18 != -8 )
        {
          v20 = v18[5];
          if ( (_QWORD *)v20 != v18 + 7 )
            _libc_free(v20);
        }
        v18 += 9;
      }
      while ( v19 != v18 );
    }
    j___libc_free_0(v27);
    return sub_15CE080(v24);
  }
  return result;
}
