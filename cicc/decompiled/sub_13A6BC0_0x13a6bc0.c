// Function: sub_13A6BC0
// Address: 0x13a6bc0
//
__int64 __fastcall sub_13A6BC0(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // rbx
  __int64 result; // rax

  if ( !a3 )
    return 1;
  v3 = a3;
  while ( 1 )
  {
    result = sub_146CEE0(*(_QWORD *)(a1 + 8), a2, v3);
    if ( !(_BYTE)result )
      break;
    v3 = (_QWORD *)*v3;
    if ( !v3 )
      return 1;
  }
  return result;
}
