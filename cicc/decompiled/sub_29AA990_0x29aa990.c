// Function: sub_29AA990
// Address: 0x29aa990
//
__int64 **__fastcall sub_29AA990(__int64 a1)
{
  __int64 v2; // r14
  __int64 i; // rbx
  __int64 **result; // rax
  __int64 v5; // rsi
  __int64 *v6; // r15
  _QWORD *v7; // r13
  __int64 *v8; // rdi
  __int64 *v9; // r15
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 *v12; // [rsp+20h] [rbp-A0h]
  __int64 *v13; // [rsp+20h] [rbp-A0h]
  __int64 v14; // [rsp+28h] [rbp-98h]
  __int64 *v15; // [rsp+30h] [rbp-90h] BYREF
  __int64 v16; // [rsp+38h] [rbp-88h]
  _BYTE v17[32]; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v18; // [rsp+60h] [rbp-60h] BYREF
  __int64 v19; // [rsp+68h] [rbp-58h]
  _BYTE v20[80]; // [rsp+70h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a1 + 80);
  v14 = a1 + 72;
  if ( a1 + 72 == v2 )
  {
    i = 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v2 + 32);
      if ( i != v2 + 24 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v14 == v2 )
        break;
      if ( !v2 )
        BUG();
    }
  }
  result = &v15;
LABEL_8:
  while ( v14 != v2 )
  {
    v5 = i - 24;
    if ( !i )
      v5 = 0;
    v18 = (__int64 *)v20;
    v15 = (__int64 *)v17;
    v16 = 0x400000000LL;
    v19 = 0x400000000LL;
    sub_AE7A50((__int64)&v15, v5, (__int64)&v18);
    v6 = v15;
    v12 = &v15[(unsigned int)v16];
    if ( v12 != v15 )
    {
      do
      {
        v7 = (_QWORD *)*v6;
        if ( a1 != sub_B43CB0(*v6) )
          sub_B43D60(v7);
        ++v6;
      }
      while ( v12 != v6 );
    }
    v8 = v18;
    v9 = v18;
    v13 = &v18[(unsigned int)v19];
    if ( v13 != v18 )
    {
      do
      {
        v10 = (_QWORD *)*v9;
        if ( a1 != sub_B141A0(*v9) )
          sub_B14290(v10);
        ++v9;
      }
      while ( v13 != v9 );
      v8 = v18;
    }
    if ( v8 != (__int64 *)v20 )
      _libc_free((unsigned __int64)v8);
    if ( v15 != (__int64 *)v17 )
      _libc_free((unsigned __int64)v15);
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v2 + 32) )
    {
      v11 = v2 - 24;
      if ( !v2 )
        v11 = 0;
      result = (__int64 **)(v11 + 48);
      if ( (__int64 **)i != result )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v14 == v2 )
        goto LABEL_8;
      if ( !v2 )
        BUG();
    }
  }
  return result;
}
