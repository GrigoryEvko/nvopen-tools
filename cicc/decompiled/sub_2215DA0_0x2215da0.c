// Function: sub_2215DA0
// Address: 0x2215da0
//
void __fastcall sub_2215DA0(__int64 *a1, size_t a2, char a3)
{
  unsigned __int64 v3; // rax

  v3 = *(_QWORD *)(*a1 - 24);
  if ( a2 > 0x3FFFFFFFFFFFFFF9LL )
    sub_4262D8((__int64)"basic_string::resize");
  if ( a2 > v3 )
  {
    sub_2215CF0(a1, a2 - v3, a3);
  }
  else if ( a2 < v3 )
  {
    sub_2215540((volatile signed __int32 **)a1, a2, v3 - a2, 0);
  }
}
