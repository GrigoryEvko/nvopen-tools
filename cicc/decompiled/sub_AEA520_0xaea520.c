// Function: sub_AEA520
// Address: 0xaea520
//
__int64 __fastcall sub_AEA520(int a1, int a2, int a3, int a4, __int64 a5, __int64 a6)
{
  char v10; // r9
  __int64 result; // rax
  int v12; // edx
  __int64 v13; // rax
  _BYTE *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-A8h]
  int v19; // [rsp+10h] [rbp-A0h]
  int v20; // [rsp+18h] [rbp-98h]
  __int64 v22; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v23; // [rsp+40h] [rbp-70h] BYREF
  __int64 v24; // [rsp+48h] [rbp-68h]
  _BYTE v25[96]; // [rsp+50h] [rbp-60h] BYREF

  v10 = sub_B59AF0(a5);
  result = 0;
  if ( !v10 )
  {
    v12 = *(_DWORD *)(a5 + 4);
    v24 = 0x600000000LL;
    v23 = v25;
    result = sub_AF4B80(*(_QWORD *)(*(_QWORD *)(a5 + 32 * (5LL - (v12 & 0x7FFFFFF))) + 24LL), &v22, &v23);
    if ( (_BYTE)result )
    {
      v20 = 8 * v22;
      if ( v23 != v25 )
        _libc_free(v23, &v22);
      v19 = sub_B595C0(a5);
      v13 = sub_B59530(a5);
      v14 = 0;
      v24 = v15;
      v23 = (_BYTE *)v13;
      if ( (_BYTE)v15 )
        v14 = v23;
      v18 = (__int64)v14;
      v16 = *(_QWORD *)(*(_QWORD *)(a5 + 32 * (2LL - (*(_DWORD *)(a5 + 4) & 0x7FFFFFF))) + 24LL);
      sub_AF47B0(&v23, *(_QWORD *)(v16 + 16), *(_QWORD *)(v16 + 24));
      v17 = 0;
      if ( v25[0] )
        v17 = v24;
      return sub_AF4D30(a1, a2, a3, a4, v19, v20, 0, v18, v17, a6, (__int64)&v23);
    }
    else if ( v23 != v25 )
    {
      _libc_free(v23, &v22);
      return 0;
    }
  }
  return result;
}
