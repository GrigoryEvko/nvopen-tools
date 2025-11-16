// Function: sub_2C1A8F0
// Address: 0x2c1a8f0
//
__int64 __fastcall sub_2C1A8F0(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rax
  _BYTE *v11; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v12[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v13; // [rsp+30h] [rbp-40h]

  v13 = 260;
  v4 = *(_QWORD *)(a2 + 904);
  v12[0] = a1 + 168;
  v5 = (_BYTE *)sub_2BFB120(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), a3);
  v6 = sub_2BFB120(a2, **(_QWORD **)(a1 + 48), a3);
  v7 = *(_QWORD **)(v4 + 72);
  v11 = v5;
  v8 = v6;
  v9 = sub_BCB2B0(v7);
  return sub_921130((unsigned int **)v4, v9, v8, &v11, 1, (__int64)v12, 0);
}
