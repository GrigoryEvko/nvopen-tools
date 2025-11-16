// Function: sub_869D30
// Address: 0x869d30
//
_BYTE *sub_869D30()
{
  _BYTE *v0; // r12
  _BYTE *v1; // rax

  v0 = 0;
  if ( !dword_4F04C3C )
  {
    v1 = sub_727090();
    v1[16] = 0;
    v0 = v1;
    sub_869970((__int64)v1);
  }
  return v0;
}
