// Function: sub_B2F990
// Address: 0xb2f990
//
_QWORD *__fastcall sub_B2F990(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  _QWORD *result; // rax

  v5 = *(_QWORD *)(a1 + 48);
  if ( v5 )
    result = sub_AA8870(v5, a1, a3, a4);
  *(_QWORD *)(a1 + 48) = a2;
  if ( a2 )
    return sub_AA8820(a2, a1);
  return result;
}
