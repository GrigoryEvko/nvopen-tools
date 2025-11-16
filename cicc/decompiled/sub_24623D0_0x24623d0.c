// Function: sub_24623D0
// Address: 0x24623d0
//
__int64 __fastcall sub_24623D0(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // r14
  __int64 v7; // r13
  __int64 *v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  unsigned __int64 v19; // r8
  __int64 v20; // rdx
  __int64 *v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  _QWORD v27[10]; // [rsp+20h] [rbp-50h] BYREF

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 12 || (unsigned int)v1 - 17 <= 1 )
    return sub_AD62B0(a1);
  if ( v1 == 16 )
  {
    v3 = sub_24623D0(*(_QWORD *)(a1 + 24));
    v6 = *(_QWORD *)(a1 + 32);
    v25 = v27;
    v7 = v3;
    v26 = 0x400000000LL;
    if ( v6 > 4 )
    {
      sub_C8D5F0((__int64)&v25, v27, v6, 8u, v4, v5);
      v23 = v25;
      v8 = &v25[v6];
      if ( v25 == v8 )
      {
LABEL_10:
        LODWORD(v26) = v6;
        v10 = sub_AD1300((__int64 **)a1, v8, (unsigned int)v6);
        v11 = (unsigned __int64)v25;
        v12 = v10;
        if ( v25 == v27 )
          return v12;
        goto LABEL_20;
      }
      do
        *v23++ = v7;
      while ( v8 != v23 );
    }
    else
    {
      v8 = v27;
      if ( !v6 )
        goto LABEL_10;
      v9 = v27;
      do
        *v9++ = v7;
      while ( &v27[v6] != v9 );
    }
    v8 = v25;
    goto LABEL_10;
  }
  if ( v1 != 15 )
    BUG();
  v13 = *(unsigned int *)(a1 + 12);
  v25 = v27;
  v26 = 0x400000000LL;
  if ( (_DWORD)v13 )
  {
    v14 = 8 * v13;
    v15 = 0;
    do
    {
      v16 = sub_24623D0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + v15));
      v18 = (unsigned int)v26;
      v19 = (unsigned int)v26 + 1LL;
      if ( v19 > HIDWORD(v26) )
      {
        v24 = v16;
        sub_C8D5F0((__int64)&v25, v27, (unsigned int)v26 + 1LL, 8u, v19, v17);
        v18 = (unsigned int)v26;
        v16 = v24;
      }
      v15 += 8;
      v25[v18] = v16;
      v20 = (unsigned int)(v26 + 1);
      LODWORD(v26) = v26 + 1;
    }
    while ( v14 != v15 );
    v21 = v25;
  }
  else
  {
    v20 = 0;
    v21 = v27;
  }
  v22 = sub_AD24A0((__int64 **)a1, v21, v20);
  v11 = (unsigned __int64)v25;
  v12 = v22;
  if ( v25 == v27 )
    return v12;
LABEL_20:
  _libc_free(v11);
  return v12;
}
