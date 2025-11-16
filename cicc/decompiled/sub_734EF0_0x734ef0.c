// Function: sub_734EF0
// Address: 0x734ef0
//
_DWORD *__fastcall sub_734EF0(__int64 a1)
{
  _DWORD *result; // rax
  __int64 v3; // rdi

  result = sub_734A70(*(_QWORD *)(a1 + 120));
  v3 = *(_QWORD *)(a1 + 256);
  if ( *(_QWORD *)(a1 + 120) != v3 )
  {
    if ( v3 )
      return sub_734A70(v3);
  }
  return result;
}
