// Function: sub_3738450
// Address: 0x3738450
//
void __fastcall sub_3738450(__int64 *a1, __int64 a2, __int64 a3, __int16 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rax
  unsigned __int64 *v12; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v13; // [rsp+28h] [rbp-C8h]
  __int64 v14[3]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE *v15; // [rsp+48h] [rbp-A8h]
  _BYTE v16[72]; // [rsp+58h] [rbp-98h] BYREF
  __int64 **v17; // [rsp+A0h] [rbp-50h]

  v8 = sub_A777F0(0x10u, a1 + 11);
  if ( v8 )
  {
    *(_QWORD *)v8 = 0;
    *(_DWORD *)(v8 + 8) = 0;
  }
  sub_3247620((__int64)v14, a1[23], (__int64)a1, v8);
  sub_3243D60(v14, a2);
  sub_32435C0((__int64)v14, (_BYTE *)a5, a2);
  v12 = 0;
  v13 = 0;
  if ( a2 )
  {
    v12 = *(unsigned __int64 **)(a2 + 16);
    v13 = *(_QWORD *)(a2 + 24);
  }
  if ( sub_AF46F0(a2) )
    sub_3243610(v14, (__int64)&v12);
  v9 = *(_QWORD *)(*(_QWORD *)(a1[23] + 232) + 16LL);
  v10 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 200LL))(v9);
  if ( (unsigned __int8)sub_3243770((__int64)v14, v10, &v12, *(_DWORD *)(a5 + 4)) )
  {
    sub_3244870(v14, &v12);
    sub_3243D40((__int64)v14);
    sub_3249620(a1, a3, a4, v17);
    if ( v16[63] )
      sub_3249A20(a1, (unsigned __int64 **)(a3 + 8), 15875, 65547, v16[62]);
  }
  if ( v15 != v16 )
    _libc_free((unsigned __int64)v15);
}
