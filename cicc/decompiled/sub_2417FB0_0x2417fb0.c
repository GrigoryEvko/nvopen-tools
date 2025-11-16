// Function: sub_2417FB0
// Address: 0x2417fb0
//
void __fastcall sub_2417FB0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // [rsp+8h] [rbp-E8h] BYREF
  _BYTE v5[32]; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v6; // [rsp+30h] [rbp-C0h]
  unsigned int *v7[2]; // [rsp+40h] [rbp-B0h] BYREF
  char v8; // [rsp+50h] [rbp-A0h] BYREF
  void *v9; // [rsp+C0h] [rbp-30h]

  sub_2417E00(a1, a2);
  if ( (_BYTE)qword_4FE3408 )
  {
    sub_23D0AB0((__int64)v7, a2, 0, 0, 0);
    v4 = sub_24159D0((__int64)*a1, a2);
    v6 = 257;
    v2 = sub_921880(v7, *(_QWORD *)(**a1 + 504), *(_QWORD *)(**a1 + 512), (int)&v4, 1, (__int64)v5, 0);
    v3 = (__int64 *)sub_BD5C60(v2);
    *(_QWORD *)(v2 + 72) = sub_A7A090((__int64 *)(v2 + 72), v3, 1, 79);
    nullsub_61();
    v9 = &unk_49DA100;
    nullsub_63();
    if ( (char *)v7[0] != &v8 )
      _libc_free((unsigned __int64)v7[0]);
  }
}
