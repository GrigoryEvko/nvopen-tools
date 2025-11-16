// Function: sub_AE8EA0
// Address: 0xae8ea0
//
void __fastcall sub_AE8EA0(__int64 a1, __int64 (__fastcall *a2)(__int64), __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v5 = sub_B91C10(a1, 18);
    if ( v5 )
    {
      v6 = sub_AE5D10(v5, a2, a3);
      sub_B99FD0(a1, 18, v6);
    }
  }
}
