// Function: sub_15D7E60
// Address: 0x15d7e60
//
void __fastcall sub_15D7E60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 *v6; // r14
  __int64 *v7; // rcx
  _QWORD *v8; // rbx
  _QWORD *v9; // r14
  unsigned __int64 v10; // rdi
  __int64 *v11; // rbx
  __int64 *v12; // r14
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15[4]; // [rsp+10h] [rbp-100h] BYREF
  _QWORD *v16; // [rsp+30h] [rbp-E0h]
  unsigned int v17; // [rsp+40h] [rbp-D0h]
  __int64 *v18; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+58h] [rbp-B8h]
  _BYTE v20[176]; // [rsp+60h] [rbp-B0h] BYREF

  v5 = sub_15CC510(a1, a2);
  if ( v5 )
  {
    *(_BYTE *)(a1 + 72) = 0;
    v6 = (__int64 *)v5;
    v7 = (__int64 *)sub_15CC510(a1, a3);
    if ( v7 )
    {
      sub_15D6FB0(a1, 0, *v6, v7);
    }
    else
    {
      v19 = 0x800000000LL;
      v18 = (__int64 *)v20;
      sub_15CDE90((__int64)v15, 0);
      sub_15D2890((__int64)v15, a3, 0, a1, (__int64)&v18, 0);
      sub_15D2F60(v15, a1, 0);
      sub_15D2160((__int64)v15, a1, v6);
      if ( v17 )
      {
        v8 = v16;
        v9 = &v16[9 * v17];
        do
        {
          if ( *v8 != -16 && *v8 != -8 )
          {
            v10 = v8[5];
            if ( (_QWORD *)v10 != v8 + 7 )
              _libc_free(v10);
          }
          v8 += 9;
        }
        while ( v9 != v8 );
      }
      j___libc_free_0(v16);
      sub_15CE080(v15);
      v11 = v18;
      v12 = &v18[2 * (unsigned int)v19];
      if ( v18 != v12 )
      {
        do
        {
          v13 = *v11;
          v11 += 2;
          v14 = (__int64 *)sub_15CC510(a1, v13);
          sub_15D6FB0(a1, 0, *v14, (__int64 *)*(v11 - 1));
        }
        while ( v12 != v11 );
        v12 = v18;
      }
      if ( v12 != (__int64 *)v20 )
        _libc_free((unsigned __int64)v12);
    }
  }
}
