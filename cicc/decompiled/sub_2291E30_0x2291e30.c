// Function: sub_2291E30
// Address: 0x2291e30
//
__int64 __fastcall sub_2291E30(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  __int64 result; // rax

  *(_BYTE *)(a5 + 43) = 0;
  result = sub_228FE90(a1, a2, a3, a5);
  if ( !(_BYTE)result )
    return sub_2291C00(a1, a2, a3, a4, a5);
  return result;
}
