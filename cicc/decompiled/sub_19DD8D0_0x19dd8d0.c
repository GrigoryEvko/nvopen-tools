// Function: sub_19DD8D0
// Address: 0x19dd8d0
//
__int64 *__fastcall sub_19DD8D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r12
  bool v7; // zf
  _BYTE v9[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v10; // [rsp+10h] [rbp-30h]

  v6 = (__int64 *)sub_19DD7C0(a1, a2, a4);
  if ( v6 )
  {
    v7 = *(_BYTE *)(a4 + 16) == 35;
    v10 = 257;
    if ( v7 )
      v6 = (__int64 *)sub_15FB440(11, v6, a3, (__int64)v9, a4);
    else
      v6 = (__int64 *)sub_15FB440(15, v6, a3, (__int64)v9, a4);
    sub_164B7C0((__int64)v6, a4);
  }
  return v6;
}
