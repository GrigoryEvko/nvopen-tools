// Function: sub_390FE40
// Address: 0x390fe40
//
_DWORD *__fastcall sub_390FE40(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  _DWORD *result; // rax

  v2 = *(_QWORD *)(a1 + 288);
  if ( a2 >= (unsigned __int64)(0x6DB6DB6DB6DB6DB7LL * ((*(_QWORD *)(a1 + 296) - v2) >> 3)) )
    return 0;
  result = (_DWORD *)(v2 + 56LL * a2);
  if ( !*result )
    return 0;
  return result;
}
