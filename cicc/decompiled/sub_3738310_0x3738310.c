// Function: sub_3738310
// Address: 0x3738310
//
void __fastcall sub_3738310(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  unsigned __int64 *v9[2]; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE v10[24]; // [rsp+20h] [rbp-C0h] BYREF
  char *v11; // [rsp+38h] [rbp-A8h]
  char v12; // [rsp+48h] [rbp-98h] BYREF
  char v13; // [rsp+84h] [rbp-5Ch]
  unsigned __int8 v14; // [rsp+86h] [rbp-5Ah]
  char v15; // [rsp+87h] [rbp-59h]
  __int64 **v16; // [rsp+90h] [rbp-50h]

  v6 = sub_A777F0(0x10u, a1 + 11);
  if ( v6 )
  {
    *(_QWORD *)v6 = 0;
    *(_DWORD *)(v6 + 8) = 0;
  }
  sub_3247620((__int64)v10, a1[23], (__int64)a1, v6);
  if ( !*(_BYTE *)a4 )
    v13 = v13 & 0xF8 | 2;
  v7 = a1[23];
  v9[0] = 0;
  v9[1] = 0;
  v8 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(v7 + 232) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(v7 + 232) + 16LL));
  if ( (unsigned __int8)sub_3243770((__int64)v10, v8, v9, *(_DWORD *)(a4 + 4)) )
  {
    sub_3244870(v10, v9);
    sub_3243D40((__int64)v10);
    sub_3249620(a1, a2, a3, v16);
    if ( v15 )
      sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 15875, 65547, v14);
  }
  if ( v11 != &v12 )
    _libc_free((unsigned __int64)v11);
}
