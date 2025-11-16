// Function: sub_6E1C20
// Address: 0x6e1c20
//
void __fastcall sub_6E1C20(_QWORD *a1, int a2, __int64 *a3)
{
  __int64 v3; // rcx
  _QWORD *v4; // rax

  v3 = *a3;
  if ( a2 )
  {
    *a3 = (__int64)a1;
    do
    {
      v4 = a1;
      a1 = (_QWORD *)*a1;
    }
    while ( a1 );
    if ( a3[1] )
      *v4 = v3;
    else
      a3[1] = (__int64)v4;
  }
  else
  {
    if ( v3 )
      *(_QWORD *)a3[1] = a1;
    else
      *a3 = (__int64)a1;
    a3[1] = (__int64)a1;
  }
}
