// Function: sub_16BDF00
// Address: 0x16bdf00
//
_QWORD *__fastcall sub_16BDF00(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r13
  __int64 *v5; // [rsp+8h] [rbp-B8h] BYREF
  unsigned __int64 v6[2]; // [rsp+10h] [rbp-B0h] BYREF
  _BYTE v7[160]; // [rsp+20h] [rbp-A0h] BYREF

  v6[1] = 0x2000000000LL;
  v2 = *a1;
  v6[0] = (unsigned __int64)v7;
  (*(void (__fastcall **)(__int64 *, __int64 *, unsigned __int64 *))(v2 + 8))(a1, a2, v6);
  v3 = sub_16BDDE0((__int64)a1, (__int64)v6, (__int64 *)&v5);
  if ( !v3 )
  {
    v3 = a2;
    sub_16BDA20(a1, a2, v5);
  }
  if ( (_BYTE *)v6[0] != v7 )
    _libc_free(v6[0]);
  return v3;
}
