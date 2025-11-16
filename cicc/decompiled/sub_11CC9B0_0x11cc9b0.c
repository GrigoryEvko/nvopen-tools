// Function: sub_11CC9B0
// Address: 0x11cc9b0
//
__int64 __fastcall sub_11CC9B0(__int64 a1, __int64 *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // rsi
  __int64 v12; // r12
  _BYTE *v14; // [rsp+0h] [rbp-80h] BYREF
  size_t v15; // [rsp+8h] [rbp-78h]
  unsigned int v16; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v17[3]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v18[72]; // [rsp+38h] [rbp-48h] BYREF

  v9 = *(_QWORD *)(a1 + 8);
  v14 = a3;
  v15 = a4;
  v17[1] = 0;
  v17[2] = 20;
  v10 = *(_BYTE *)(v9 + 8) == 3;
  v17[0] = (__int64)v18;
  if ( !v10 )
    sub_11C53C0(a1, (__int64)&v14, v17, a4, a5, (__int64)a6);
  sub_980AF0(*a2, v14, v15, &v16);
  v11 = v16;
  v12 = sub_11CC8D0(a1, v16, (__int64)v14, v15, a5, a6, a2);
  if ( (_BYTE *)v17[0] != v18 )
    _libc_free(v17[0], v11);
  return v12;
}
