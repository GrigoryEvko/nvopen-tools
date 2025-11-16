// Function: sub_FDB000
// Address: 0xfdb000
//
__int64 __fastcall sub_FDB000(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_BB9660(a2, (__int64)&unk_4F8E808);
  result = sub_BB9660(a2, (__int64)&unk_4F875EC);
  *(_BYTE *)(a2 + 160) = 1;
  return result;
}
