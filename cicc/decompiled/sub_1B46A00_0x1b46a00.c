// Function: sub_1B46A00
// Address: 0x1b46a00
//
__int64 __fastcall sub_1B46A00(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // rdx

  v2 = ((*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1) - 1;
  result = sub_1B44DF0(a1, 0, a1, v2, a2);
  if ( v2 == v4 )
    return a1;
  return result;
}
