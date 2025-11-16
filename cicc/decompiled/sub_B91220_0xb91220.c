// Function: sub_B91220
// Address: 0xb91220
//
void __fastcall sub_B91220(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax

  v2 = sub_B911C0((unsigned __int8 *)a2);
  if ( v2 )
  {
    sub_B90FD0((__int64)v2, a1);
  }
  else if ( *(_BYTE *)a2 == 3 )
  {
    *(_QWORD *)(a2 + 8) = 0;
  }
}
