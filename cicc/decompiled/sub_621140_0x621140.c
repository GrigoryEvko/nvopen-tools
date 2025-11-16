// Function: sub_621140
// Address: 0x621140
//
_BOOL8 __fastcall sub_621140(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // rbx
  int v4; // r13d
  int v5; // r15d
  int v6; // r8d
  _BOOL8 result; // rax

  v3 = 16LL * a3;
  v4 = byte_4B6DF90[a3];
  v5 = sub_620E90(a1);
  v6 = sub_621000((__int16 *)(a1 + 176), v5, (__int16 *)((char *)&unk_4F067A0 + v3), v4);
  result = 0;
  if ( v6 >= 0 )
  {
    if ( a1 != a2 )
      v5 = sub_620E90(a2);
    return (int)sub_621000((__int16 *)(a2 + 176), v5, (__int16 *)(v3 + 82863808), v4) <= 0;
  }
  return result;
}
