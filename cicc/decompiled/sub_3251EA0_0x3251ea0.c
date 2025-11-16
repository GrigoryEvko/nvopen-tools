// Function: sub_3251EA0
// Address: 0x3251ea0
//
__int64 __fastcall sub_3251EA0(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12[4]; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  sub_31F0A90(*(_QWORD *)(*(_QWORD *)a1 + 8LL), **(_DWORD **)(a1 + 8));
  if ( **(_BYTE **)(a1 + 16) )
  {
    v9 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v14 = 1;
    v12[0] = (__int64)"ttbaseref";
    v13 = 3;
    v10 = sub_31DCC50(v9, v12, v2, v3, v4);
    sub_31DCA60(*(_QWORD *)(*(_QWORD *)a1 + 8LL));
    v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 224LL);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v11 + 208LL))(v11, v10, 0);
  }
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  v14 = 1;
  v12[0] = (__int64)"cst_begin";
  v13 = 3;
  v6 = sub_31DCC50(v5, v12, v2, v3, v4);
  sub_31F0A90(*(_QWORD *)(*(_QWORD *)a1 + 8LL), **(_DWORD **)(a1 + 32));
  sub_31DCA60(*(_QWORD *)(*(_QWORD *)a1 + 8LL));
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 224LL);
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 208LL))(v7, v6, 0);
}
