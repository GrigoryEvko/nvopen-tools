// Function: sub_2216A40
// Address: 0x2216a40
//
void __fastcall sub_2216A40(__int64 *a1, size_t a2, wchar_t a3)
{
  unsigned __int64 v3; // rax

  v3 = *(_QWORD *)(*a1 - 24);
  if ( a2 > 0xFFFFFFFFFFFFFFELL )
    sub_4262D8((__int64)"basic_string::resize");
  if ( a2 > v3 )
  {
    sub_2216980(a1, a2 - v3, a3);
  }
  else if ( a2 < v3 )
  {
    sub_22161B0((const wchar_t **)a1, a2, v3 - a2, 0);
  }
}
