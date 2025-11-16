// Function: sub_164B400
// Address: 0x164b400
//
__int64 __fastcall sub_164B400(__int64 a1)
{
  unsigned __int64 v1; // rax

  v1 = sub_16498B0(a1);
  if ( v1 )
    _libc_free(v1);
  return sub_164B0D0(a1, 0);
}
