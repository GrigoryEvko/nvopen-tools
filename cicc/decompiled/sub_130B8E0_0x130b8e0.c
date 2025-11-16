// Function: sub_130B8E0
// Address: 0x130b8e0
//
__int64 __fastcall sub_130B8E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)result )
    return sub_13488C0(a1, a2 + 62384);
  return result;
}
