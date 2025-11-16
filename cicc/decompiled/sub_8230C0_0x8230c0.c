// Function: sub_8230C0
// Address: 0x8230c0
//
__int64 __fastcall sub_8230C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d

  if ( dword_4F073A0 == 0x7FFFFFFF )
    sub_685240(0x8Fu);
  v6 = dword_4F073A0 + 1;
  dword_4F073A0 = v6;
  sub_823040(v6, a2, a3, a4, a5, a6);
  return v6;
}
