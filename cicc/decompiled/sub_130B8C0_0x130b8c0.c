// Function: sub_130B8C0
// Address: 0x130b8c0
//
__int64 __fastcall sub_130B8C0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)result )
    return sub_1348810(a1, a2 + 62384, a3);
  return result;
}
