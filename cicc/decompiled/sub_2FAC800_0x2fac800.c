// Function: sub_2FAC800
// Address: 0x2fac800
//
__int64 __fastcall sub_2FAC800(__int64 a1, __int64 a2)
{
  _QWORD **v2; // r13
  __int64 v3; // r15
  __int64 *v4; // rax
  unsigned __int64 v5; // rax
  __int64 **v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h]
  _QWORD v21[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = *(_QWORD ***)(a2 + 40);
  v3 = sub_BCE3C0(**(__int64 ***)(a1 + 24), 0);
  v4 = (__int64 *)sub_BCB120(*v2);
  v21[0] = v3;
  v19 = v21;
  v20 = 0x100000001LL;
  v5 = sub_BCF480(v4, v21, 1, 0);
  *(_QWORD *)(a1 + 32) = sub_BA8C10((__int64)v2, (__int64)"_Unwind_SjLj_Register", 0x15u, v5, 0);
  v6 = *(__int64 ***)(a1 + 24);
  *(_QWORD *)(a1 + 40) = v7;
  v8 = sub_BCE3C0(*v6, 0);
  v9 = (__int64 *)sub_BCB120(*v2);
  v21[0] = v8;
  v19 = v21;
  v20 = 0x100000001LL;
  v10 = sub_BCF480(v9, v21, 1, 0);
  v11 = sub_BA8C10((__int64)v2, (__int64)"_Unwind_SjLj_Unregister", 0x17u, v10, 0);
  *(_QWORD *)(a1 + 56) = v12;
  *(_QWORD *)(a1 + 48) = v11;
  v13 = sub_BCE3C0(*v2, *((_DWORD *)v2 + 79));
  v19 = (_QWORD *)v13;
  *(_QWORD *)(a1 + 72) = sub_B6E160((__int64 *)v2, 0xB2u, (__int64)&v19, 1);
  v19 = (_QWORD *)v13;
  *(_QWORD *)(a1 + 80) = sub_B6E160((__int64 *)v2, 0x157u, (__int64)&v19, 1);
  v19 = (_QWORD *)v13;
  *(_QWORD *)(a1 + 88) = sub_B6E160((__int64 *)v2, 0x156u, (__int64)&v19, 1);
  *(_QWORD *)(a1 + 64) = sub_B6E160((__int64 *)v2, 0x55u, 0, 0);
  *(_QWORD *)(a1 + 96) = sub_B6E160((__int64 *)v2, 0x53u, 0, 0);
  *(_QWORD *)(a1 + 104) = sub_B6E160((__int64 *)v2, 0x50u, 0, 0);
  *(_QWORD *)(a1 + 112) = sub_B6E160((__int64 *)v2, 0x51u, 0, 0);
  return sub_2FA9880((__int64 *)a1, a2, v14, v15, v16, v17);
}
