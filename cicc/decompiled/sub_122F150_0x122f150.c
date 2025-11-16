// Function: sub_122F150
// Address: 0x122f150
//
__int64 __fastcall sub_122F150(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 a4, unsigned __int8 a5)
{
  __int64 result; // rax

  result = sub_120AFE0(a1, 12, "expected '(' in call");
  if ( !(_BYTE)result )
    return sub_122ED50(a1, a2, a3, a4, a5);
  return result;
}
