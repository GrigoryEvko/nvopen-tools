// Function: sub_2CD3AE0
// Address: 0x2cd3ae0
//
void __fastcall sub_2CD3AE0(_BYTE *a1, _QWORD *a2, __int64 a3, unsigned __int64 a4, __int64 *a5)
{
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r13
  int v12; // edx
  __int64 v13; // rax
  __int64 **v15; // [rsp+10h] [rbp-120h]
  _QWORD v17[4]; // [rsp+20h] [rbp-110h] BYREF
  _QWORD v18[2]; // [rsp+40h] [rbp-F0h] BYREF
  _QWORD v19[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v20; // [rsp+60h] [rbp-D0h]
  unsigned int *v21[2]; // [rsp+70h] [rbp-C0h] BYREF
  char v22; // [rsp+80h] [rbp-B0h] BYREF
  void *v23; // [rsp+F0h] [rbp-40h]

  sub_23D0AB0((__int64)v21, (__int64)a2, 0, 0, 0);
  v6 = (__int64 *)a2[1];
  v15 = *(__int64 ***)(*(_QWORD *)(a2[5] + 72LL) + 40LL);
  v7 = sub_AC8EA0(*v15, a5);
  v8 = *(a2 - 8);
  v17[2] = v7;
  v17[0] = v8;
  v9 = *(a2 - 4);
  v18[0] = v19;
  v17[1] = v9;
  v19[0] = v6;
  v19[1] = v6;
  v20 = v6;
  v18[1] = 0x300000003LL;
  v10 = sub_BCF480(v6, v19, 3, 0);
  v11 = sub_BA8C10((__int64)v15, a3, a4, v10, 0);
  LOWORD(v20) = 257;
  v13 = sub_921880(v21, v11, v12, (int)v17, 3, (__int64)v18, 0);
  sub_BD84D0((__int64)a2, v13);
  sub_B43D60(a2);
  *a1 = 1;
  nullsub_61();
  v23 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v21[0] != &v22 )
    _libc_free((unsigned __int64)v21[0]);
}
