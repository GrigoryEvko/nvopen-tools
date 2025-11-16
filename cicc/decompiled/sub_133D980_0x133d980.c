// Function: sub_133D980
// Address: 0x133d980
//
unsigned __int64 __fastcall sub_133D980(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax

  v4 = sub_130B0E0(a1 + 128);
  v5 = sub_130B0E0(a2) / v4;
  if ( v5 <= 0xC7 )
    return (unsigned __int64)(((_QWORD)&loc_1000000 - qword_42878A0[199 - v5]) * a3) >> 24;
  return a3;
}
