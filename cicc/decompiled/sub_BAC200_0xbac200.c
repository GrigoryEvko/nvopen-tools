// Function: sub_BAC200
// Address: 0xbac200
//
void __fastcall sub_BAC200(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  _QWORD v6[2]; // [rsp-38h] [rbp-38h] BYREF
  __int64 v7; // [rsp-28h] [rbp-28h] BYREF

  if ( *(_BYTE *)(a2 + 32) > 1u )
  {
    v4 = *(_QWORD *)(a1 + 32);
    if ( v4 )
    {
      if ( v4 == 0x3FFFFFFFFFFFFFFFLL || v4 == 4611686018427387902LL )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(a1 + 24, ", ", 2, a4);
    }
    else
    {
      sub_2241130(a1 + 24, 0, 0, " // ", 4);
    }
    sub_CA0F50(v6, a2);
    sub_2241490(a1 + 24, v6[0], v6[1], v5);
    if ( (__int64 *)v6[0] != &v7 )
      j_j___libc_free_0(v6[0], v7 + 1);
  }
}
