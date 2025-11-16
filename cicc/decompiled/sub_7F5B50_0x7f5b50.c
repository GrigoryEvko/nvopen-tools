// Function: sub_7F5B50
// Address: 0x7f5b50
//
_BYTE *__fastcall sub_7F5B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5)
{
  _QWORD *v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _BYTE *result; // rax
  __int64 v15; // r13
  __int64 *v16; // rsi
  __int64 v18; // rdi

  if ( a3 )
  {
    v8 = sub_73E830(a3);
    v9 = sub_73DCD0(v8);
    result = sub_731370((__int64)v9, a2, v10, v11, v12, v13);
    v15 = (__int64)result;
    if ( !result )
      return result;
  }
  else
  {
    result = *(_BYTE **)(a1 + 168);
    v18 = *((_QWORD *)result + 24);
    if ( !v18 )
      return result;
    result = sub_7F5690(v18, a1, 0);
    v15 = (__int64)result;
    if ( !result )
      return result;
  }
  v16 = 0;
  if ( !a4 )
    v16 = (__int64 *)sub_7E2530(a2);
  return sub_7F5940(v15, v16, a4, a5);
}
