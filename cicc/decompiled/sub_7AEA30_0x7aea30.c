// Function: sub_7AEA30
// Address: 0x7aea30
//
__int64 __fastcall sub_7AEA30(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax

  *(_QWORD *)(a3 + 8) = a1;
  if ( a1 )
  {
    do
    {
      v3 = a1;
      a1 = (_QWORD *)*a1;
    }
    while ( a1 );
    *(_QWORD *)(a3 + 16) = v3;
    return sub_7AE210(a3);
  }
  else
  {
    *(_QWORD *)(a3 + 16) = 0;
    return sub_7AE210(a3);
  }
}
