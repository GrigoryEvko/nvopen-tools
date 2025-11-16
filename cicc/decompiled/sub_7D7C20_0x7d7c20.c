// Function: sub_7D7C20
// Address: 0x7d7c20
//
_BYTE *__fastcall sub_7D7C20(__int64 a1)
{
  __int64 *v1; // r12
  _QWORD *v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rax

  v1 = (__int64 *)sub_7D7B30(a1);
  v2 = sub_73A830(1, unk_4F06A60);
  v3 = *v1;
  v1[2] = (__int64)v2;
  v4 = sub_8D46C0(v3);
  return sub_73DC30(0x5Cu, v4, (__int64)v1);
}
