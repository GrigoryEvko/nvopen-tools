// Function: sub_740870
// Address: 0x740870
//
__int64 __fastcall sub_740870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // r12
  __int64 **v10; // rbx

  if ( a1 )
  {
    result = (__int64)sub_73E730(a1, a2, a3, a4, a5, a6);
    v9 = result;
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 56);
    a2 = 0;
    result = sub_731770(v9, 0, a3, a4, a5, a6);
    if ( !(_DWORD)result )
      return result;
  }
  if ( v9 )
  {
    v10 = (__int64 **)(a4 + 56);
    if ( a3 )
      v10 = (__int64 **)sub_740760(a3, a2);
    result = (__int64)sub_73DF90(v9, *v10);
    *v10 = (__int64 *)result;
  }
  return result;
}
