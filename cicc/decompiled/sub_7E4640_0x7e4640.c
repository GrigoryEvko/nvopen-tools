// Function: sub_7E4640
// Address: 0x7e4640
//
void *__fastcall sub_7E4640(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // r12
  _QWORD *v8; // rax

  v2 = sub_7E45A0(a1);
  v7 = sub_731370((__int64)v2, a2, v3, v4, v5, v6);
  v7[2] = sub_73A830(a2, unk_4F06A60);
  v8 = sub_7E1330();
  return sub_73DBF0(0x5Cu, (__int64)v8, (__int64)v7);
}
