// Function: sub_38A1070
// Address: 0x38a1070
//
__int64 __fastcall sub_38A1070(__int64 **a1, __int64 a2, _QWORD *a3, __int64 *a4, double a5, double a6, double a7)
{
  unsigned int v8; // r12d
  unsigned int v10; // eax
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int16 *v15; // [rsp+18h] [rbp-108h]
  __int64 v16[4]; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v17; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v18; // [rsp+58h] [rbp-C8h]
  __int64 v19; // [rsp+68h] [rbp-B8h]
  _QWORD *v20; // [rsp+70h] [rbp-B0h]
  __int64 v21; // [rsp+78h] [rbp-A8h]
  _BYTE v22[16]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD *v23; // [rsp+90h] [rbp-90h]
  __int64 v24; // [rsp+98h] [rbp-88h]
  _BYTE v25[16]; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v26; // [rsp+B0h] [rbp-70h]
  unsigned int v27; // [rsp+B8h] [rbp-68h]
  char v28; // [rsp+BCh] [rbp-64h]
  void *v29; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v30; // [rsp+D0h] [rbp-50h]
  unsigned __int64 v31; // [rsp+E8h] [rbp-38h]

  *a3 = 0;
  v20 = v22;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v21 = 0;
  v22[0] = 0;
  v23 = v25;
  v24 = 0;
  v25[0] = 0;
  v27 = 1;
  v26 = 0;
  v28 = 0;
  v15 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v16, 0.0);
  sub_169E320(&v29, v16, v15);
  sub_1698460((__int64)v16);
  v31 = 0;
  v8 = sub_389C540((__int64)a1, (__int64)&v17, 0.0, a6, a7);
  if ( !(_BYTE)v8 )
  {
    LOBYTE(v10) = sub_389BAC0(a1, a2, &v17, a3, a4, 0);
    v8 = v10;
  }
  if ( v31 )
    j_j___libc_free_0_0(v31);
  if ( v29 == sub_16982C0() )
  {
    v11 = v30;
    if ( v30 )
    {
      v12 = 32LL * *(_QWORD *)(v30 - 8);
      v13 = v30 + v12;
      if ( v30 != v30 + v12 )
      {
        do
        {
          v13 -= 32;
          sub_127D120((_QWORD *)(v13 + 8));
        }
        while ( v11 != v13 );
      }
      j_j_j___libc_free_0_0(v11 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v29);
  }
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v23 != (_QWORD *)v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  if ( v20 != (_QWORD *)v22 )
    j_j___libc_free_0((unsigned __int64)v20);
  return v8;
}
