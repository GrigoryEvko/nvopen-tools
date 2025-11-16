// Function: sub_1683B30
// Address: 0x1683b30
//
__int64 __fastcall sub_1683B30(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      v1 = (_QWORD *)*v1;
      result = sub_16856A0(v2);
    }
    while ( v1 );
  }
  return result;
}
