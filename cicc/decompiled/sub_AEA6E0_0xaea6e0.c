// Function: sub_AEA6E0
// Address: 0xaea6e0
//
__int64 __fastcall sub_AEA6E0(int a1, int a2, int a3, int a4, __int64 a5, __int64 a6)
{
  char v9; // r8
  __int64 result; // rax
  __int64 v11; // rax
  int v12; // r8d
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // [rsp+8h] [rbp-A8h]
  int v18; // [rsp+10h] [rbp-A0h]
  _QWORD v21[2]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v22; // [rsp+40h] [rbp-70h] BYREF
  __int64 v23; // [rsp+48h] [rbp-68h]
  _BYTE v24[96]; // [rsp+50h] [rbp-60h] BYREF

  v9 = sub_B14070(a5);
  result = 0;
  if ( !v9 )
  {
    v22 = v24;
    v23 = 0x600000000LL;
    v11 = sub_B11F60(a5 + 88);
    result = sub_AF4B80(v11, v21, &v22);
    if ( (_BYTE)result )
    {
      v18 = 8 * LODWORD(v21[0]);
      if ( v22 != v24 )
        _libc_free(v22, v21);
      v17 = sub_B13320(a5);
      sub_B12FD0(&v22, a5);
      v12 = v17;
      if ( v24[0] )
      {
        v13 = (__int64)v22;
        v14 = v23;
      }
      else
      {
        v15 = sub_B13000(a5);
        v13 = 0;
        v12 = v17;
        v21[1] = v16;
        v21[0] = v15;
        if ( (_BYTE)v16 )
          v13 = v21[0];
        v14 = 0;
      }
      return sub_AF4D30(a1, a2, a3, a4, v12, v18, 0, v13, v14, a6, (__int64)v21);
    }
    else if ( v22 != v24 )
    {
      _libc_free(v22, v21);
      return 0;
    }
  }
  return result;
}
