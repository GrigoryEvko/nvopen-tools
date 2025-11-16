// Function: sub_15F3B70
// Address: 0x15f3b70
//
void __fastcall sub_15F3B70(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // [rsp+8h] [rbp-38h] BYREF
  _DWORD *v4; // [rsp+10h] [rbp-30h]
  __int64 v5; // [rsp+18h] [rbp-28h]
  _DWORD v6[8]; // [rsp+20h] [rbp-20h] BYREF

  v6[0] = a2;
  v4 = v6;
  v5 = 0x100000001LL;
  v3 = sub_16498A0(a1);
  v2 = sub_161BD30(&v3, v6, 1);
  sub_1625C10(a1, 2, v2);
  if ( v4 != v6 )
    _libc_free((unsigned __int64)v4);
}
