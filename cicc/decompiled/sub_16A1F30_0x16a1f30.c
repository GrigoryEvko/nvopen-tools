// Function: sub_16A1F30
// Address: 0x16a1f30
//
__int64 __fastcall sub_16A1F30(__int64 a1, char a2, double a3, double a4, double a5)
{
  __int16 *v7; // rax
  __int16 *v8; // rbx
  __int64 v9; // rax
  __int16 *v10; // rdx
  __int16 **v11; // r15
  void **v12; // rdi
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // r13
  __int16 **v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int16 *v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-58h]
  __int16 *v27; // [rsp+38h] [rbp-48h] BYREF
  __int64 v28; // [rsp+40h] [rbp-40h]

  v26 = 64;
  v25 = 0x360000000000000LL;
  v7 = (__int16 *)sub_16982C0();
  v8 = v7;
  if ( v7 == word_42AE9D0 )
    sub_169D060(&v27, (__int64)v7, &v25);
  else
    sub_169D050((__int64)&v27, word_42AE9D0, &v25);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = v27;
  v11 = (__int16 **)(v9 + 8);
  if ( *(__int16 **)(v9 + 8) != v8 )
  {
    if ( v8 != v27 )
    {
      sub_16983E0(v9 + 8, (__int64)&v27);
      goto LABEL_6;
    }
    if ( v11 == &v27 )
      goto LABEL_17;
    goto LABEL_15;
  }
  if ( v8 != v27 )
  {
    if ( v11 == &v27 )
      goto LABEL_7;
LABEL_15:
    sub_127D120((_QWORD *)(v9 + 8));
    if ( v8 != v27 )
    {
      sub_1698450((__int64)v11, (__int64)&v27);
      if ( v27 != v8 )
        goto LABEL_7;
      goto LABEL_17;
    }
    goto LABEL_43;
  }
  if ( v11 == &v27 )
    goto LABEL_17;
  v18 = *(_QWORD *)(v9 + 16);
  if ( v18 )
  {
    v19 = 32LL * *(_QWORD *)(v18 - 8);
    v20 = v18 + v19;
    while ( v18 != v20 )
    {
      v21 = v20 - 32;
      v22 = v10;
      v23 = v18;
      v24 = v21;
      if ( *(__int16 **)(v21 + 8) == v10 )
      {
        sub_169DEB0((__int64 *)(v21 + 16));
        v10 = v22;
        v18 = v23;
        v20 = v24;
      }
      else
      {
        sub_1698460(v21 + 8);
        v20 = v24;
        v18 = v23;
        v10 = v22;
      }
    }
    j_j_j___libc_free_0_0(v18 - 8);
  }
LABEL_43:
  sub_169C7E0(v11, &v27);
LABEL_6:
  if ( v27 != v8 )
  {
LABEL_7:
    sub_1698460((__int64)&v27);
    goto LABEL_8;
  }
LABEL_17:
  v14 = v28;
  if ( v28 )
  {
    v15 = 32LL * *(_QWORD *)(v28 - 8);
    v16 = v28 + v15;
    if ( v28 != v28 + v15 )
    {
      do
      {
        v16 -= 32;
        if ( v8 == *(__int16 **)(v16 + 8) )
          sub_169DEB0((__int64 *)(v16 + 16));
        else
          sub_1698460(v16 + 8);
      }
      while ( v14 != v16 );
    }
    j_j_j___libc_free_0_0(v14 - 8);
  }
LABEL_8:
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( a2 )
  {
    v17 = (__int16 **)(*(_QWORD *)(a1 + 8) + 8LL);
    if ( v8 == *v17 )
      sub_169C8D0((__int64)v17, a3, a4, a5);
    else
      sub_1699490((__int64)v17);
  }
  v12 = (void **)(*(_QWORD *)(a1 + 8) + 40LL);
  if ( v8 == *v12 )
    return sub_169C980(v12, 0);
  else
    return sub_169B620((__int64)v12, 0);
}
