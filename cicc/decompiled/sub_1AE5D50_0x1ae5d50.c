// Function: sub_1AE5D50
// Address: 0x1ae5d50
//
__int64 __fastcall sub_1AE5D50(__int64 a1, unsigned __int16 a2, float a3)
{
  _QWORD *v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  __int64 ***v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int16 *v14; // [rsp+0h] [rbp-D0h]
  __int64 v15; // [rsp+0h] [rbp-D0h]
  __int64 v16; // [rsp+8h] [rbp-C8h]
  __int64 v17[4]; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int8 *v18; // [rsp+30h] [rbp-A0h] BYREF
  void *v19; // [rsp+38h] [rbp-98h] BYREF
  __int64 v20; // [rsp+40h] [rbp-90h]
  __int64 v21[3]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v22; // [rsp+68h] [rbp-68h]
  __int64 v23; // [rsp+70h] [rbp-60h]
  int v24; // [rsp+78h] [rbp-58h]
  __int64 v25; // [rsp+80h] [rbp-50h]
  __int64 v26; // [rsp+88h] [rbp-48h]

  v4 = (_QWORD *)sub_16498A0(a1);
  v5 = *(unsigned __int8 **)(a1 + 48);
  v21[0] = 0;
  v22 = v4;
  v6 = *(_QWORD *)(a1 + 40);
  v23 = 0;
  v21[1] = v6;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v21[2] = a1 + 24;
  v18 = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)&v18, (__int64)v5, 2);
    if ( v21[0] )
      sub_161E7C0((__int64)v21, v21[0]);
    v21[0] = (__int64)v18;
    if ( v18 )
      sub_1623210((__int64)&v18, v18, (__int64)v21);
  }
  v7 = *(__int64 ****)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v14 = (__int16 *)sub_1698270();
  sub_169D3B0((__int64)v17, (__m128i)LODWORD(a3));
  sub_169E320(&v19, v17, v14);
  sub_1698460((__int64)v17);
  v8 = sub_159CCF0(v22, (__int64)&v18);
  if ( v19 == sub_16982C0() )
  {
    v11 = v20;
    if ( v20 )
    {
      v12 = 32LL * *(_QWORD *)(v20 - 8);
      v13 = v20 + v12;
      if ( v20 != v20 + v12 )
      {
        do
        {
          v15 = v11;
          v16 = v13 - 32;
          sub_127D120((_QWORD *)(v13 - 24));
          v13 = v16;
          v11 = v15;
        }
        while ( v15 != v16 );
      }
      j_j_j___libc_free_0_0(v11 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v19);
  }
  if ( *((_BYTE *)*v7 + 8) != 2 )
    v8 = sub_15A3E10(v8, *v7, 0);
  LOWORD(v20) = 257;
  v9 = sub_1289B20(v21, a2, v7, v8, (__int64)&v18, 0);
  if ( v21[0] )
    sub_161E7C0((__int64)v21, v21[0]);
  return v9;
}
