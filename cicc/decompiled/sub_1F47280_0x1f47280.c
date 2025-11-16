// Function: sub_1F47280
// Address: 0x1f47280
//
__int64 __fastcall sub_1F47280(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 (*v3)(); // rax

  sub_1F46F00(a1, &unk_4FC3334, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC9074, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC4534, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC6A0C, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC8A0C, 0, 1, 0);
  if ( byte_4FCC1C0 )
    sub_1F46F00(a1, &unk_4FC450C, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FCE24C, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC9D8C, 1, 1, 0);
  sub_1F46F00(a1, &unk_4FCA200, 1, 1, 0);
  result = sub_1F46F00(a1, &unk_4FC7874, 1, 1, 0);
  if ( a2 )
  {
    sub_1F46490(a1, a2, 1, 1, 0);
    v3 = *(__int64 (**)())(*(_QWORD *)a1 + 336LL);
    if ( v3 != sub_1F44660 )
      ((void (__fastcall *)(__int64))v3)(a1);
    sub_1F46F00(a1, &unk_4FCE41C, 1, 1, 0);
    sub_1F46F00(a1, &unk_4FCAC8C, 1, 1, 0);
    sub_1F46F00(a1, &unk_4FC5C8C, 1, 1, 0);
    return sub_1F46F00(a1, &unk_4FC64C9, 1, 1, 0);
  }
  return result;
}
