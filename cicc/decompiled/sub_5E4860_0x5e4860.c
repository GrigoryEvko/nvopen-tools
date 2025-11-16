// Function: sub_5E4860
// Address: 0x5e4860
//
_QWORD *__fastcall sub_5E4860(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax
  unsigned __int16 v3; // cx
  _QWORD *v4; // rdi

  result = *(_QWORD **)(a1 + 120);
  if ( result && (v3 = *(_WORD *)(a2[2] + 224LL), v3 >= *(_WORD *)(result[2] + 224LL)) )
  {
    do
    {
      v4 = result;
      result = (_QWORD *)*result;
    }
    while ( result && v3 >= *(_WORD *)(result[2] + 224LL) );
    *a2 = result;
    *v4 = a2;
  }
  else
  {
    *(_QWORD *)(a1 + 120) = a2;
    *a2 = result;
  }
  return result;
}
