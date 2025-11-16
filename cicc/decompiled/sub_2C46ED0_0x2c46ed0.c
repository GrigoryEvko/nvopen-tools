// Function: sub_2C46ED0
// Address: 0x2c46ed0
//
__int64 *__fastcall sub_2C46ED0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( !sub_2BF04A0(a1) )
    return sub_DD8400(a2, *(_QWORD *)(a1 + 40));
  v2 = sub_2BF0490(a1);
  if ( *(_BYTE *)(v2 + 8) == 2 )
    return *(__int64 **)(v2 + 152);
  else
    return (__int64 *)sub_D970F0(a2);
}
