// Function: sub_6F41B0
// Address: 0x6f41b0
//
void __fastcall sub_6F41B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  while ( a1 )
  {
    sub_6F40D0((__int64)a1, a2, a3, a4, a5, a6);
    if ( !*a1 )
      break;
    if ( *(_BYTE *)(*a1 + 8LL) == 3 )
      a1 = (_QWORD *)sub_6BBB10(a1);
    else
      a1 = (_QWORD *)*a1;
  }
}
