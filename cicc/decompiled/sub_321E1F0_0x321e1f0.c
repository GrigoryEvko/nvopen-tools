// Function: sub_321E1F0
// Address: 0x321e1f0
//
void __fastcall sub_321E1F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // [rsp+0h] [rbp-60h] BYREF
  char *v3; // [rsp+8h] [rbp-58h]
  char v4; // [rsp+18h] [rbp-48h] BYREF

  sub_321B470((__int64)&v2, a2);
  sub_321E150(a1, &v2);
  if ( v3 != &v4 )
    _libc_free((unsigned __int64)v3);
}
