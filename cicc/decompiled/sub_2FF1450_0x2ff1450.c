// Function: sub_2FF1450
// Address: 0x2ff1450
//
__int64 __fastcall sub_2FF1450(__int64 a1)
{
  __int64 result; // rax
  void (*v2)(); // rax

  sub_2FF12A0(a1, &unk_501CF54, 0);
  sub_2FF12A0(a1, &unk_503BCAC, 0);
  sub_2FF12A0(a1, &unk_503FCF4, 0);
  sub_2FF12A0(a1, &unk_502A64C, 0);
  sub_2FF12A0(a1, &unk_501EB14, 0);
  sub_2FF12A0(a1, &unk_50208AC, 0);
  sub_2FF12A0(a1, &unk_5022C2C, 0);
  if ( byte_5027B68 )
    sub_2FF12A0(a1, &unk_501EACC, 0);
  sub_2FF12A0(a1, &unk_502A48C, 0);
  sub_2FF12A0(a1, &unk_502476C, 0);
  sub_2FF12A0(a1, &unk_5024F60, 0);
  sub_2FF12A0(a1, &unk_5021074, 0);
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 440LL))(a1);
  if ( (_BYTE)result )
  {
    sub_2FF12A0(a1, &unk_502624C, 0);
    v2 = *(void (**)())(*(_QWORD *)a1 + 352LL);
    if ( v2 != nullsub_1693 )
      ((void (__fastcall *)(__int64))v2)(a1);
    sub_2FF12A0(a1, &unk_501F390, 0);
    return sub_2FF12A0(a1, &unk_50201E9, 0);
  }
  return result;
}
