// Function: sub_389C160
// Address: 0x389c160
//
__int64 __fastcall sub_389C160(__int64 **a1, __int64 a2, _QWORD *a3)
{
  __int16 *v3; // r12
  unsigned int v4; // eax
  _QWORD *v5; // r10
  unsigned int v6; // r12d
  unsigned int v8; // eax
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v13; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v14[2]; // [rsp+30h] [rbp-F0h] BYREF
  char v15; // [rsp+40h] [rbp-E0h]
  char v16; // [rsp+41h] [rbp-DFh]
  unsigned int v17; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int64 v18; // [rsp+58h] [rbp-C8h]
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
  v3 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v14, 0.0);
  sub_169E320(&v29, v14, v3);
  sub_1698460((__int64)v14);
  v31 = 0;
  v13 = 0;
  v4 = sub_389C540(a1, &v17, 0);
  v5 = a3;
  v6 = v4;
  if ( !(_BYTE)v4 )
  {
    LOBYTE(v8) = sub_389BAC0(a1, a2, &v17, &v13, 0, 0);
    v5 = a3;
    v6 = v8;
  }
  if ( v13 )
  {
    if ( *(_BYTE *)(v13 + 16) > 0x10u )
    {
      *v5 = 0;
      v16 = 1;
      v14[0] = (__int64)"global values must be constants";
      v15 = 3;
      v6 = sub_38814C0((__int64)(a1 + 1), v18, (__int64)v14);
    }
    else
    {
      *v5 = v13;
    }
  }
  if ( v31 )
    j_j___libc_free_0_0(v31);
  if ( v29 == sub_16982C0() )
  {
    v9 = v30;
    if ( v30 )
    {
      v10 = 32LL * *(_QWORD *)(v30 - 8);
      v11 = v30 + v10;
      if ( v30 != v30 + v10 )
      {
        do
        {
          v11 -= 32;
          sub_127D120((_QWORD *)(v11 + 8));
        }
        while ( v9 != v11 );
      }
      j_j_j___libc_free_0_0(v9 - 8);
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
  return v6;
}
