// Function: sub_35979B0
// Address: 0x35979b0
//
__int64 __fastcall sub_35979B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_BYTE *)(a2 + 160) = 1;
  result = sub_BB9660(a2, (__int64)&unk_5025C1C);
  *(_BYTE *)(a2 + 160) = 1;
  return result;
}
