// Function: sub_77F6E0
// Address: 0x77f6e0
//
_QWORD *__fastcall sub_77F6E0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx

  v1 = *(_QWORD **)(a1 + 200);
  if ( !v1 )
    return (_QWORD *)(a1 + 200);
  do
  {
    v2 = v1;
    v1 = (_QWORD *)*v1;
  }
  while ( v1 );
  return v2;
}
