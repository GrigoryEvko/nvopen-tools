// Function: sub_23049C0
// Address: 0x23049c0
//
__int64 *__fastcall sub_23049C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  int v10; // [rsp+4h] [rbp-6Ch]
  __int64 v11; // [rsp+8h] [rbp-68h]
  _QWORD v12[5]; // [rsp+10h] [rbp-60h] BYREF
  int v13; // [rsp+38h] [rbp-38h]

  sub_228C9B0(v12, a2 + 8, a3, a4);
  v4 = v12[0];
  v5 = v12[1];
  v10 = v13;
  v6 = v12[2];
  v11 = v12[3];
  v7 = v12[4];
  v8 = sub_22077B0(0x38u);
  if ( v8 )
  {
    *(_QWORD *)(v8 + 40) = v7;
    *(_QWORD *)(v8 + 8) = v4;
    *(_QWORD *)(v8 + 16) = v5;
    *(_QWORD *)v8 = &unk_4A0B0D8;
    *(_QWORD *)(v8 + 24) = v6;
    *(_QWORD *)(v8 + 32) = v11;
    *(_DWORD *)(v8 + 48) = v10;
  }
  *a1 = v8;
  return a1;
}
