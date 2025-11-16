// Function: sub_33D00B0
// Address: 0x33d00b0
//
_QWORD *__fastcall sub_33D00B0(
        __int64 a1,
        int a2,
        unsigned __int64 a3,
        int a4,
        unsigned __int64 *a5,
        __int64 a6,
        int a7)
{
  _QWORD *v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v10; // [rsp+8h] [rbp-C8h] BYREF
  __int64 v11; // [rsp+10h] [rbp-C0h] BYREF
  int v12; // [rsp+18h] [rbp-B8h]
  _QWORD v13[2]; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE v14[160]; // [rsp+30h] [rbp-A0h] BYREF

  v7 = 0;
  if ( *(_WORD *)(a3 + 16LL * (unsigned int)(a4 - 1)) != 262 )
  {
    v13[1] = 0x2000000000LL;
    v13[0] = v14;
    sub_33C9670((__int64)v13, a2, a3, a5, a6, a6);
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v7 = sub_33CCCF0(a1, (__int64)v13, (__int64)&v11, &v10);
    if ( v11 )
      sub_B91220((__int64)&v11, v11);
    if ( !v7 )
    {
      v8 = v13[0];
      if ( (_BYTE *)v13[0] == v14 )
        return v7;
      goto LABEL_6;
    }
    sub_33D00A0((__int64)v7, a7);
    v8 = v13[0];
    if ( (_BYTE *)v13[0] != v14 )
LABEL_6:
      _libc_free(v8);
  }
  return v7;
}
