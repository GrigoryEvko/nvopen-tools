// Function: sub_86A190
// Address: 0x86a190
//
__int64 __fastcall sub_86A190(_QWORD **a1, _QWORD *a2)
{
  _QWORD *v2; // rbx
  _QWORD *i; // r12
  __int64 result; // rax

  v2 = a1;
  for ( i = *a1; ; i = (_QWORD *)*i )
  {
    result = sub_86A080(v2);
    if ( v2 == a2 )
      break;
    v2 = i;
  }
  return result;
}
