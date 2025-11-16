// Function: sub_E81C50
// Address: 0xe81c50
//
__int64 __fastcall sub_E81C50(__int64 a1)
{
  unsigned __int64 v1; // rsi
  __int64 v2; // rdi
  __int64 result; // rax

  v1 = *(unsigned int *)(a1 + 24);
  v2 = *(_QWORD *)(a1 + 8);
  result = 0;
  if ( *(unsigned int *)(v2 + 184) > v1 )
    return *(_QWORD *)(*(_QWORD *)(v2 + 176) + 8 * v1);
  return result;
}
