// Function: sub_15D7CB0
// Address: 0x15d7cb0
//
void __fastcall sub_15D7CB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 *v8; // r15
  __int64 *v9; // rcx
  _QWORD *v10; // rbx
  _QWORD *v11; // r15
  unsigned __int64 v12; // rdi
  __int64 *v13; // rbx
  __int64 *v14; // r14
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17[4]; // [rsp+10h] [rbp-100h] BYREF
  _QWORD *v18; // [rsp+30h] [rbp-E0h]
  unsigned int v19; // [rsp+40h] [rbp-D0h]
  __int64 *v20; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+58h] [rbp-B8h]
  _BYTE v22[176]; // [rsp+60h] [rbp-B0h] BYREF

  v7 = sub_15CC510(a1, a3);
  if ( v7 )
  {
    *(_BYTE *)(a1 + 72) = 0;
    v8 = (__int64 *)v7;
    v9 = (__int64 *)sub_15CC510(a1, a4);
    if ( v9 )
    {
      sub_15D6FB0(a1, a2, *v8, v9);
    }
    else
    {
      v20 = (__int64 *)v22;
      v21 = 0x800000000LL;
      sub_15CDE90((__int64)v17, a2);
      sub_15D2890((__int64)v17, a4, 0, a1, (__int64)&v20, 0);
      sub_15D2F60(v17, a1, 0);
      sub_15D2160((__int64)v17, a1, v8);
      if ( v19 )
      {
        v10 = v18;
        v11 = &v18[9 * v19];
        do
        {
          if ( *v10 != -16 && *v10 != -8 )
          {
            v12 = v10[5];
            if ( (_QWORD *)v12 != v10 + 7 )
              _libc_free(v12);
          }
          v10 += 9;
        }
        while ( v11 != v10 );
      }
      j___libc_free_0(v18);
      sub_15CE080(v17);
      v13 = v20;
      v14 = &v20[2 * (unsigned int)v21];
      if ( v20 != v14 )
      {
        do
        {
          v15 = *v13;
          v13 += 2;
          v16 = (__int64 *)sub_15CC510(a1, v15);
          sub_15D6FB0(a1, a2, *v16, (__int64 *)*(v13 - 1));
        }
        while ( v14 != v13 );
        v14 = v20;
      }
      if ( v14 != (__int64 *)v22 )
        _libc_free((unsigned __int64)v14);
    }
  }
}
