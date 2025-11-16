// Function: sub_6EA0A0
// Address: 0x6ea0a0
//
__int64 __fastcall sub_6EA0A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 result; // rax

  v2 = (__int64 *)sub_73A930();
  result = sub_6E7150(v2, a2);
  *(_BYTE *)(a2 + 20) |= 1u;
  return result;
}
