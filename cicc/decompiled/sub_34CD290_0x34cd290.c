// Function: sub_34CD290
// Address: 0x34cd290
//
_QWORD *__fastcall sub_34CD290(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  _QWORD *v5; // rax
  _QWORD v7[5]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v8; // [rsp+28h] [rbp-38h]

  sub_34BE2E0(v7, a2, a3);
  v3 = v7[2];
  v4 = v7[3];
  v8 = v7[1];
  v5 = (_QWORD *)sub_22077B0(0x28u);
  if ( v5 )
  {
    v5[3] = v3;
    v5[4] = v4;
    *v5 = &unk_4A37B10;
    v5[2] = v8;
    v5[1] = &unk_4A37808;
  }
  *a1 = v5;
  return a1;
}
