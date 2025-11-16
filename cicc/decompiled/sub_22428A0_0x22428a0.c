// Function: sub_22428A0
// Address: 0x22428a0
//
__int64 *__fastcall sub_22428A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  wchar_t *v3; // rbp
  __int64 v4; // rsi

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(wchar_t **)(v2 + 64);
  if ( !v3 )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v4 = (__int64)&v3[wcslen(*(const wchar_t **)(v2 + 64))];
  if ( v3 == (wchar_t *)v4 )
    *a1 = (__int64)&unk_4FD67F8;
  else
    *a1 = sub_2242590(v3, v4);
  return a1;
}
