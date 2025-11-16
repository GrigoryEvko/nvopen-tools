// Function: sub_C65340
// Address: 0xc65340
//
__int64 __fastcall sub_C65340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  result = malloc(a1, a2, a3, a4, a5, a6);
  if ( !result && (a1 || (result = malloc(1, a2, v7, v8, v9, v10)) == 0) )
    sub_C64F00("Allocation failed", 1u);
  return result;
}
