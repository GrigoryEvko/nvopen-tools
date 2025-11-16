// Function: sub_3738190
// Address: 0x3738190
//
void __fastcall sub_3738190(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 i; // r12
  unsigned int v7; // r14d
  unsigned __int64 *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rax
  __int64 v12; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v13[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v14[3]; // [rsp+30h] [rbp-C0h] BYREF
  char *v15; // [rsp+48h] [rbp-A8h]
  char v16; // [rsp+58h] [rbp-98h] BYREF
  __int64 **v17; // [rsp+A0h] [rbp-50h]

  v5 = sub_A777F0(0x10u, a1 + 11);
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_DWORD *)(v5 + 8) = 0;
  }
  sub_3247620((__int64)v14, a1[23], (__int64)a1, v5);
  for ( i = *(_QWORD *)(a2 + 24); a2 + 8 != i; i = sub_220EF30(i) )
  {
    v7 = *(_DWORD *)(i + 32);
    v12 = *(_QWORD *)(i + 40);
    sub_3243D60(v14, v12);
    v8 = *(unsigned __int64 **)(v12 + 24);
    v13[0] = *(unsigned __int64 **)(v12 + 16);
    v13[1] = v8;
    sub_3243610(v14, (__int64)v13);
    v9 = *(_QWORD *)(*(_QWORD *)(a1[23] + 232) + 16LL);
    v10 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 200LL))(v9);
    sub_3243770((__int64)v14, v10, v13, v7);
    sub_3244870(v14, v13);
  }
  sub_3243D40((__int64)v14);
  sub_3249620(a1, a4, 2, v17);
  if ( v15 != &v16 )
    _libc_free((unsigned __int64)v15);
}
