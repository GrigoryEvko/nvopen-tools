// Function: sub_16E8C20
// Address: 0x16e8c20
//
void *__fastcall sub_16E8C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // [rsp+0h] [rbp-10h] BYREF
  __int64 v7; // [rsp+8h] [rbp-8h]

  v6 = 0;
  v7 = sub_2241E40(a1, a2, a3, a4, a5);
  if ( byte_4FA1770 || !(unsigned int)sub_2207590(&byte_4FA1770) )
    return &unk_4FA1780;
  sub_16E8AF0((__int64)&unk_4FA1780, "-", 1, (__int64)&v6, 0);
  __cxa_atexit((void (*)(void *))sub_16E7C30, &unk_4FA1780, &qword_4A427C0);
  sub_2207640(&byte_4FA1770);
  return &unk_4FA1780;
}
