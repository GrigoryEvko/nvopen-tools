// Function: sub_F92F70
// Address: 0xf92f70
//
void __fastcall sub_F92F70(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rax

  if ( *(_BYTE *)(a1 + 64) )
  {
    v4 = *(_QWORD *)a1;
    v5 = sub_B53F50(a1);
    a2 = 2;
    sub_B99FD0(v4, 2u, v5);
  }
  if ( *(_BYTE *)(a1 + 56) )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( v3 != a1 + 24 )
      _libc_free(v3, a2);
  }
}
