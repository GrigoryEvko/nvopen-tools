// Function: sub_341C200
// Address: 0x341c200
//
__int64 (__fastcall *__fastcall sub_341C200(__int64 a1, __int64 a2, __int64 a3))(_QWORD *, _QWORD *, __int64)
{
  __int64 *v3; // rbx
  _QWORD *v4; // r15
  _QWORD *v5; // r8
  __int64 (*v6)(void); // rax
  __int64 v8; // rax
  __int64 v9; // [rsp-8h] [rbp-D8h]
  _QWORD *v10; // [rsp+0h] [rbp-D0h]
  __int64 v11; // [rsp+8h] [rbp-C8h]
  _QWORD *v12; // [rsp+8h] [rbp-C8h]
  __int64 v13; // [rsp+18h] [rbp-B8h] BYREF
  _QWORD *v14; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v15; // [rsp+28h] [rbp-A8h]
  _QWORD v16[20]; // [rsp+30h] [rbp-A0h] BYREF

  if ( a3 )
  {
    v3 = *(__int64 **)(a3 + 40);
    v4 = 0;
    v5 = *(_QWORD **)(a3 + 64);
    v6 = *(__int64 (**)(void))(*(_QWORD *)v3[2] + 128LL);
    if ( v6 != sub_2DAC790 )
    {
      v12 = *(_QWORD **)(a3 + 64);
      v8 = v6();
      v5 = v12;
      v4 = (_QWORD *)v8;
    }
    v10 = v5;
    v11 = v3[6];
    sub_A558A0((__int64)v16, *(_QWORD *)(*v3 + 40), 1);
    sub_A564B0((__int64)v16, *v3);
    v14 = v16;
    v15 = 0;
    sub_2EAC530(a2, a1, (__int64)v16, (__int64)&v14, v10, v11, v4);
    if ( v14 != v16 )
      _libc_free((unsigned __int64)v14);
    return sub_A55520(v16, a1);
  }
  else
  {
    sub_B6EEA0(&v13);
    sub_A558A0((__int64)v16, 0, 1);
    v14 = v16;
    v15 = 0;
    sub_2EAC530(a2, a1, (__int64)v16, (__int64)&v14, &v13, 0, 0);
    if ( v14 != v16 )
      _libc_free((unsigned __int64)v14);
    sub_A55520(v16, v9);
    return (__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))sub_B6E710(&v13);
  }
}
