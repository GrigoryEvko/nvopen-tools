// Function: sub_20941C0
// Address: 0x20941c0
//
__int64 __fastcall sub_20941C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int64 v4; // r13
  _QWORD *v5; // r8
  __int64 (*v6)(void); // rax
  __int64 v8; // rax
  _QWORD *v9; // [rsp+0h] [rbp-90h]
  __int64 v10; // [rsp+8h] [rbp-88h]
  _QWORD *v11; // [rsp+8h] [rbp-88h]
  __int64 v12; // [rsp+18h] [rbp-78h] BYREF
  __int64 *v13; // [rsp+20h] [rbp-70h] BYREF
  __int64 v14; // [rsp+28h] [rbp-68h]
  __int64 v15[12]; // [rsp+30h] [rbp-60h] BYREF

  if ( a3 )
  {
    v3 = *(__int64 **)(a3 + 32);
    v4 = 0;
    v5 = *(_QWORD **)(a3 + 48);
    v6 = *(__int64 (**)(void))(*(_QWORD *)v3[2] + 40LL);
    if ( v6 != sub_1D00B00 )
    {
      v11 = *(_QWORD **)(a3 + 48);
      v8 = v6();
      v5 = v11;
      v4 = v8;
    }
    v9 = v5;
    v10 = v3[7];
    sub_154BA10((__int64)v15, *(_QWORD *)(*v3 + 40), 1);
    sub_154C150((__int64)v15, *v3);
    v13 = v15;
    v14 = 0;
    sub_1E343B0(a2, a1, (__int64)v15, (__int64)&v13, v9, v10, v4);
    if ( v13 != v15 )
      _libc_free((unsigned __int64)v13);
    return sub_154BA40(v15);
  }
  else
  {
    sub_1602D10(&v12);
    sub_154BA10((__int64)v15, 0, 1);
    v13 = v15;
    v14 = 0;
    sub_1E343B0(a2, a1, (__int64)v15, (__int64)&v13, &v12, 0, 0);
    if ( v13 != v15 )
      _libc_free((unsigned __int64)v13);
    sub_154BA40(v15);
    return sub_16025D0(&v12);
  }
}
