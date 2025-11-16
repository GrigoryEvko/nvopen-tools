// Function: sub_FEE180
// Address: 0xfee180
//
__int64 __fastcall sub_FEE180(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  result = sub_BB9660(a2, (__int64)&unk_4F8FBD4);
  *(_BYTE *)(a2 + 160) = 1;
  return result;
}
