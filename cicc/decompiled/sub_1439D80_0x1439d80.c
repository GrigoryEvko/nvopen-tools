// Function: sub_1439D80
// Address: 0x1439d80
//
__int64 __fastcall sub_1439D80(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_13E6840(a2);
  *(_BYTE *)(a2 + 160) = 1;
  return result;
}
