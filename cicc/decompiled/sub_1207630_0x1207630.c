// Function: sub_1207630
// Address: 0x1207630
//
__int64 *__fastcall sub_1207630(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD v4[2]; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v6[6]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v7; // [rsp+50h] [rbp-40h]

  v6[5] = 0x100000000LL;
  v7 = v4;
  LOBYTE(v5[0]) = 0;
  v4[0] = v5;
  v4[1] = 0;
  memset(&v6[1], 0, 32);
  v6[0] = &unk_49DD210;
  sub_CB5980((__int64)v6, 0, 0, 0);
  sub_A587F0(a2, (__int64)v6, 0, 0);
  v2 = v7;
  *a1 = (__int64)(a1 + 2);
  sub_12060D0(a1, (_BYTE *)*v2, *v2 + v2[1]);
  v6[0] = &unk_49DD210;
  sub_CB5840((__int64)v6);
  if ( (_QWORD *)v4[0] != v5 )
    j_j___libc_free_0(v4[0], v5[0] + 1LL);
  return a1;
}
