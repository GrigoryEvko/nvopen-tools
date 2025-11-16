// Function: sub_2FA8740
// Address: 0x2fa8740
//
__int64 __fastcall sub_2FA8740(__int64 **a1, __int64 **a2)
{
  __int64 v4; // rax
  __int64 *v5; // rdi
  unsigned int v6; // esi
  __int64 *v7; // r12
  __int64 (*v8)(); // rax
  __int64 *v9; // rax
  __int64 *v10; // rax
  __int64 *v11; // rdx
  __int64 *v12; // rcx
  _QWORD *v13; // rdi
  _QWORD v15[10]; // [rsp+0h] [rbp-50h] BYREF

  v4 = sub_BCE3C0(*a2, 0);
  v5 = a1[16];
  v6 = 32;
  v7 = (__int64 *)v4;
  if ( v5 )
  {
    v8 = *(__int64 (**)())(*v5 + 160);
    if ( v8 != sub_23CE340 )
      v6 = ((__int64 (__fastcall *)(__int64 *, __int64))v8)(v5, 32);
  }
  v9 = (__int64 *)sub_BCD140(*a2, v6);
  *a1 = v9;
  a1[1] = sub_BCD420(v9, 4);
  v10 = sub_BCD420(v7, 5);
  v11 = a1[1];
  v12 = *a1;
  a1[2] = v10;
  v13 = (_QWORD *)*v7;
  v15[1] = v12;
  v15[2] = v11;
  v15[0] = v7;
  v15[3] = v7;
  v15[4] = v7;
  v15[5] = v10;
  a1[3] = sub_BD0B90(v13, v15, 6, 0);
  return 0;
}
