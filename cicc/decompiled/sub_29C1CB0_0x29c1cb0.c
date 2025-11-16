// Function: sub_29C1CB0
// Address: 0x29c1cb0
//
__int64 __fastcall sub_29C1CB0(_QWORD *a1)
{
  char v1; // r15
  _QWORD *v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // rsi
  char v6; // r12
  _BYTE *v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 *v10; // rbx
  unsigned int v11; // eax
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r14d
  __int64 v15; // r15
  __int64 *v16; // rbx
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  unsigned __int8 v22; // al
  unsigned __int64 v23; // rdi
  unsigned __int8 v25; // [rsp+Fh] [rbp-61h]
  __int64 *v26; // [rsp+10h] [rbp-60h] BYREF
  __int64 v27; // [rsp+18h] [rbp-58h]
  _BYTE v28[80]; // [rsp+20h] [rbp-50h] BYREF

  v1 = 0;
  v3 = (_QWORD *)sub_BA8DC0((__int64)a1, (__int64)"llvm.debugify", 13);
  if ( v3 )
  {
    v1 = 1;
    sub_BA9050((__int64)a1, v3);
  }
  v5 = (_QWORD *)sub_BA8DC0((__int64)a1, (__int64)"llvm.mir.debugify", 17);
  if ( v5 )
  {
    v1 = 1;
    sub_BA9050((__int64)a1, v5);
  }
  v6 = sub_AEB840(a1, (__int64)v5, v4);
  v7 = sub_BA8CB0((__int64)a1, (__int64)"llvm.dbg.value", 0xEu);
  if ( v7 )
  {
    sub_B2E860(v7);
    v25 = 1;
  }
  else
  {
    v25 = v6 | v1;
  }
  v8 = a1[108];
  if ( v8 )
  {
    v9 = a1[108];
    v10 = (__int64 *)v28;
    v11 = sub_B91A00(v9);
    v14 = v11;
    v26 = (__int64 *)v28;
    v27 = 0x400000000LL;
    if ( v11 )
    {
      if ( v11 > 4 )
      {
        sub_C8D5F0((__int64)&v26, v28, v11, 8u, v12, v13);
        v10 = &v26[(unsigned int)v27];
      }
      v15 = 0;
      do
      {
        v10[v15] = sub_B91A10(v8, v15);
        ++v15;
      }
      while ( v14 != v15 );
      v14 += v27;
    }
    LODWORD(v27) = v14;
    sub_B91A30(v8);
    v16 = &v26[(unsigned int)v27];
    if ( v16 != v26 )
    {
      v17 = v26;
      do
      {
        v21 = *v17;
        v22 = *(_BYTE *)(*v17 - 16);
        if ( (v22 & 2) != 0 )
          v18 = *(_QWORD *)(v21 - 32);
        else
          v18 = v21 + -16 - 8LL * ((v22 >> 2) & 0xF);
        v19 = sub_B91420(*(_QWORD *)(v18 + 8));
        if ( v20 == 18
          && !(*(_QWORD *)v19 ^ 0x6E49206775626544LL | *(_QWORD *)(v19 + 8) ^ 0x6973726556206F66LL)
          && *(_WORD *)(v19 + 16) == 28271 )
        {
          v25 = 1;
        }
        else
        {
          sub_B979A0(v8, v21);
        }
        ++v17;
      }
      while ( v16 != v17 );
    }
    if ( !(unsigned int)sub_B91A00(v8) )
    {
      sub_B91A20(v8);
      v23 = (unsigned __int64)v26;
      if ( v26 == (__int64 *)v28 )
        return v25;
      goto LABEL_26;
    }
    v23 = (unsigned __int64)v26;
    if ( v26 != (__int64 *)v28 )
LABEL_26:
      _libc_free(v23);
  }
  return v25;
}
