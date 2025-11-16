// Function: sub_1441D10
// Address: 0x1441d10
//
char __fastcall sub_1441D10(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  int v4; // edx

  if ( !a2 )
    return 0;
  if ( (unsigned __int8)sub_1441AE0(a1) )
  {
    v3 = sub_15E44B0(a2);
    if ( v4 )
      return sub_1441CD0((__int64)a1, v3);
  }
  return 0;
}
