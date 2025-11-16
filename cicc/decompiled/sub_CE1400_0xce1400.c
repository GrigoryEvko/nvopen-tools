// Function: sub_CE1400
// Address: 0xce1400
//
__int64 __fastcall sub_CE1400(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_BB9660(a2, (__int64)&unk_4F875EC);
  result = sub_BB9660(a2, (__int64)&unk_4F8144C);
  *(_BYTE *)(a2 + 160) = 1;
  return result;
}
