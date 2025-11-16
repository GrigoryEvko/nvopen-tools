// Function: sub_334C6A0
// Address: 0x334c6a0
//
__int64 __fastcall sub_334C6A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 *v8; // rax
  __int64 *v9; // rdx
  __int64 v11[6]; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v12; // [rsp+30h] [rbp-C0h] BYREF
  __int64 *v13; // [rsp+38h] [rbp-B8h]
  __int64 v14; // [rsp+40h] [rbp-B0h]
  int v15; // [rsp+48h] [rbp-A8h]
  unsigned __int8 v16; // [rsp+4Ch] [rbp-A4h]
  __int64 v17; // [rsp+50h] [rbp-A0h] BYREF

  v13 = &v17;
  v5 = *a1;
  v11[2] = (__int64)a1;
  v11[0] = v5;
  v6 = a1[2];
  v11[3] = (__int64)&v12;
  v11[1] = v6;
  v11[4] = a3;
  v14 = 0x100000010LL;
  v15 = 0;
  v16 = 1;
  v17 = a2;
  v12 = 1;
  sub_3349730(v11, a2, a3, a4, a5);
  v7 = v16;
  if ( v16 )
  {
    v8 = v13;
    v9 = &v13[HIDWORD(v14)];
    if ( v13 == v9 )
    {
      return 0;
    }
    else
    {
      while ( a2 != *v8 )
      {
        if ( v9 == ++v8 )
          return 0;
      }
    }
  }
  else
  {
    LOBYTE(v7) = sub_C8CA60((__int64)&v12, a2) != 0;
    if ( !v16 )
      _libc_free((unsigned __int64)v13);
  }
  return v7;
}
