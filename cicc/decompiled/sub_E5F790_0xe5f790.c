// Function: sub_E5F790
// Address: 0xe5f790
//
_DWORD *__fastcall sub_E5F790(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  _DWORD *result; // rax

  v2 = *(_QWORD *)(a1 + 280);
  if ( a2 >= (unsigned __int64)(0x6DB6DB6DB6DB6DB7LL * ((*(_QWORD *)(a1 + 288) - v2) >> 3)) )
    return 0;
  result = (_DWORD *)(v2 + 56LL * a2);
  if ( !*result )
    return 0;
  return result;
}
