// Function: sub_F0BC20
// Address: 0xf0bc20
//
__int64 __fastcall sub_F0BC20(unsigned int a1, __int64 a2, char a3)
{
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // r9
  unsigned __int64 v9; // r12
  __int64 *v10; // rdi
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 *i; // rdx
  __int64 j; // r15
  unsigned __int8 *v15; // rax
  __int64 v16; // r12
  __int64 *v18; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+18h] [rbp-B8h]
  _QWORD v20[22]; // [rsp+20h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(v5 + 24);
  v7 = (__int64)sub_AD93D0(a1, v6, a3, 0);
  if ( !v7 )
  {
    if ( a3 )
    {
      if ( a1 <= 0x17 )
      {
        if ( a1 > 0x15 )
        {
          v7 = sub_AD64C0(v6, 1, 0);
          goto LABEL_2;
        }
      }
      else if ( a1 == 24 )
      {
        v7 = (__int64)sub_AD8DD0(v6, 1.0);
        goto LABEL_2;
      }
LABEL_21:
      BUG();
    }
    if ( a1 <= 0x10 )
    {
      if ( a1 <= 0xE )
        BUG();
    }
    else if ( a1 - 19 > 8 )
    {
      goto LABEL_21;
    }
    v7 = sub_AD6530(v6, v6);
  }
LABEL_2:
  v9 = *(unsigned int *)(v5 + 32);
  v18 = v20;
  v10 = v20;
  v19 = 0x1000000000LL;
  v11 = v9;
  if ( !v9 )
    goto LABEL_13;
  v12 = v20;
  if ( v9 > 0x10 )
  {
    sub_C8D5F0((__int64)&v18, v20, v9, 8u, (__int64)&v18, v8);
    v12 = &v18[(unsigned int)v19];
    for ( i = &v18[v9]; i != v12; ++v12 )
    {
LABEL_5:
      if ( v12 )
        *v12 = 0;
    }
  }
  else
  {
    i = &v20[v9];
    if ( i != v20 )
      goto LABEL_5;
  }
  LODWORD(v19) = v9;
  for ( j = 0; j != v9; ++j )
  {
    v15 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)a2, (unsigned int)j);
    if ( (unsigned int)*v15 - 12 <= 1 )
      v15 = (unsigned __int8 *)v7;
    v18[j] = (__int64)v15;
  }
  v10 = v18;
  v11 = (unsigned int)v19;
LABEL_13:
  v16 = sub_AD3730(v10, v11);
  if ( v18 != v20 )
    _libc_free(v18, v11);
  return v16;
}
