// Function: sub_228C9B0
// Address: 0x228c9b0
//
_QWORD *__fastcall sub_228C9B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax

  v6 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v7 = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  *a1 = v6 + 8;
  a1[1] = v7;
  a1[2] = v8 + 8;
  a1[3] = a3;
  return a1;
}
