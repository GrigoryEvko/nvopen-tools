// Function: sub_130C610
// Address: 0x130c610
//
__int64 __fastcall sub_130C610(__int64 a1, __int64 a2, __int64 a3, volatile signed __int64 *a4, __int64 a5, char a6)
{
  __int64 v8; // rbx
  __int64 result; // rax
  unsigned __int64 v10; // rbx
  __int64 v11; // [rsp-10h] [rbp-50h]

  v8 = sub_13427E0(a5 + 112);
  result = sub_13427E0(a5 + 9768);
  v10 = result + v8;
  if ( !*(_BYTE *)(a3 + 112) )
  {
    if ( v10 )
    {
      sub_130C320(a1, a2, a3, a4, a5, a6, 0, v10);
      return v11;
    }
  }
  return result;
}
