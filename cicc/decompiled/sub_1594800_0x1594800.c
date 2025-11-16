// Function: sub_1594800
// Address: 0x1594800
//
__int64 __fastcall sub_1594800(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r13
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rbx
  unsigned __int8 v18; // [rsp+Fh] [rbp-41h] BYREF
  _BYTE v19[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20; // [rsp+18h] [rbp-38h] BYREF
  __int64 v21; // [rsp+20h] [rbp-30h]

  v5 = sub_16982C0(a1, a2, a3, a4);
  v6 = a2 + 8;
  v7 = v5;
  if ( *(_QWORD *)(a2 + 8) == v5 )
    sub_169C6E0(&v20, v6);
  else
    sub_16986C0(&v20, v6);
  v10 = v20;
  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 1:
      v11 = 1;
      v14 = sub_1698260(&v20, v6, v8, v9);
      if ( v14 == v10 )
        goto LABEL_8;
      goto LABEL_12;
    case 2:
      v11 = 1;
      v14 = sub_1698270(&v20, v6);
      if ( v14 == v10 )
        goto LABEL_8;
      goto LABEL_12;
    case 3:
      if ( v10 == sub_1698260(&v20, v6, v8, v9) || v10 == sub_1698270(&v20, v6) )
      {
        v11 = 1;
      }
      else
      {
        v11 = 1;
        v14 = sub_1698280(&v20);
        if ( v14 != v10 )
        {
LABEL_12:
          sub_16A3360(v19, v14, 0, &v18);
          v11 = v18 ^ 1;
          if ( v7 != v20 )
            goto LABEL_9;
          goto LABEL_13;
        }
      }
LABEL_8:
      if ( v7 != v10 )
      {
LABEL_9:
        sub_1698460(&v20);
        return v11;
      }
LABEL_13:
      v15 = v21;
      if ( !v21 )
        return v11;
      v16 = 32LL * *(_QWORD *)(v21 - 8);
      v17 = v21 + v16;
      if ( v21 != v21 + v16 )
      {
        do
        {
          v17 -= 32;
          sub_127D120((_QWORD *)(v17 + 8));
        }
        while ( v15 != v17 );
      }
      j_j_j___libc_free_0_0(v15 - 8);
      return v11;
    case 4:
      v11 = 1;
      if ( v10 != sub_1698260(&v20, v6, v8, v9) && v10 != sub_1698270(&v20, v6) && v10 != sub_1698280(&v20) )
        LOBYTE(v11) = v10 == sub_16982A0();
      goto LABEL_8;
    case 5:
      v11 = 1;
      if ( v10 != sub_1698260(&v20, v6, v8, v9) && v10 != sub_1698270(&v20, v6) && v10 != sub_1698280(&v20) )
        LOBYTE(v11) = v10 == sub_1698290();
      goto LABEL_8;
    case 6:
      v11 = 1;
      if ( v10 != sub_1698260(&v20, v6, v8, v9) && v10 != sub_1698270(&v20, v6) )
      {
        v12 = sub_1698280(&v20);
        LOBYTE(v11) = v10 == v7;
        LOBYTE(v12) = v10 == v12;
        v11 |= v12;
      }
      goto LABEL_8;
    default:
      v11 = 0;
      goto LABEL_8;
  }
}
