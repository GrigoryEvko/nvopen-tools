// Function: sub_23CE0C0
// Address: 0x23ce0c0
//
void *__fastcall sub_23CE0C0(__int64 *a1)
{
  _QWORD *v2; // [rsp+0h] [rbp-80h] BYREF
  __int16 v3; // [rsp+20h] [rbp-60h]
  _QWORD v4[4]; // [rsp+30h] [rbp-50h] BYREF
  char v5; // [rsp+50h] [rbp-30h]
  _QWORD v6[2]; // [rsp+58h] [rbp-28h] BYREF
  _QWORD *v7; // [rsp+68h] [rbp-18h] BYREF

  v4[2] = &v7;
  v4[0] = "llvm-worker-{0}";
  v2 = v4;
  v6[0] = &unk_4A16298;
  v6[1] = a1 + 1;
  v7 = v6;
  v3 = 263;
  v4[1] = 15;
  v4[3] = 1;
  v5 = 1;
  sub_C95A00((__int64)&v2);
  nullsub_166();
  sub_23CDA10(*a1, 0);
  j_j___libc_free_0((unsigned __int64)a1);
  return 0;
}
