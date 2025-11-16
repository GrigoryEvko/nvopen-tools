// Function: sub_2BFB0D0
// Address: 0x2bfb0d0
//
bool __fastcall sub_2BFB0D0(__int64 a1)
{
  __int64 v1; // rbx
  bool result; // al
  __int64 *v3; // rax
  __int64 v4; // r8

  v1 = sub_2BF04A0(a1);
  result = 1;
  if ( v1 )
  {
    v3 = (__int64 *)sub_2BFACB0(*(_QWORD *)(v1 + 80));
    v4 = sub_2BF5D50(v3);
    result = 0;
    if ( v4 )
      return sub_2BF0A00(*(_QWORD *)(v1 + 80)) == 0;
  }
  return result;
}
