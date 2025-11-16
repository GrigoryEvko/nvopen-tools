// Function: sub_7AE1D0
// Address: 0x7ae1d0
//
__int64 __fastcall sub_7AE1D0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (__int64)v1;
      v1 = (_QWORD *)*v1;
      result = sub_7AD730(v2, 1);
    }
    while ( v1 );
  }
  return result;
}
