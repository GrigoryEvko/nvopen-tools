// Function: sub_1683BE0
// Address: 0x1683be0
//
__int64 __fastcall sub_1683BE0(_QWORD *a1, __int64 (__fastcall *a2)(_QWORD, __int64), __int64 a3)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // rax
  __int64 result; // rax

  if ( a1 )
  {
    v4 = a1;
    do
    {
      v5 = v4;
      v4 = (_QWORD *)*v4;
      result = a2(v5[1], a3);
    }
    while ( v4 );
  }
  return result;
}
