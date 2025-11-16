// Function: sub_B2E730
// Address: 0xb2e730
//
void __fastcall sub_B2E730(__int64 a1)
{
  __int64 v1; // rax

  if ( (*(_BYTE *)(a1 + 3) & 0x40) != 0 )
  {
    v1 = sub_B2BE50(a1);
    sub_B6F830(v1, a1);
    sub_B2E700(a1, 14, 0);
  }
}
