// Function: sub_39FAC40
// Address: 0x39fac40
//
unsigned __int64 __fastcall sub_39FAC40(unsigned __int64 a1)
{
  unsigned __int64 v1; // rdi

  v1 = ((a1 - ((a1 >> 1) & 0x5555555555555555LL)) & 0x3333333333333333LL)
     + (((a1 - ((a1 >> 1) & 0x5555555555555555LL)) >> 2) & 0x3333333333333333LL);
  return (0x101010101010101LL * ((v1 + (v1 >> 4)) & 0xF0F0F0F0F0F0F0FLL)) >> 56;
}
