// Function: sub_11CD090
// Address: 0x11cd090
//
__int64 __fastcall sub_11CD090(__int64 a1, __int64 a2, __int64 *a3, _BYTE *a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r12
  _BYTE *v13; // [rsp+0h] [rbp-80h] BYREF
  size_t v14; // [rsp+8h] [rbp-78h]
  unsigned int v15; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v16[3]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v17[72]; // [rsp+38h] [rbp-48h] BYREF

  v9 = *(_QWORD *)(a1 + 8);
  v13 = a4;
  v14 = a5;
  v16[1] = 0;
  v16[2] = 20;
  v10 = *(_BYTE *)(v9 + 8) == 3;
  v16[0] = (__int64)v17;
  if ( !v10 )
    sub_11C53C0(a1, (__int64)&v13, v16, (__int64)a4, a5, a6);
  sub_980AF0(*a3, v13, v14, &v15);
  v11 = sub_11CCF70(a1, a2, v15, (__int64)v13, v14, a6, a7, a3);
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0], a2);
  return v11;
}
