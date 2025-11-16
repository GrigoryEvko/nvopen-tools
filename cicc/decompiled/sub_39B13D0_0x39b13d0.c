// Function: sub_39B13D0
// Address: 0x39b13d0
//
__int64 __fastcall sub_39B13D0(__int64 *a1, __int64 a2, char a3, bool *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  bool v9; // r8
  bool v10; // al
  void (__fastcall *v12)(__int64, __int64, _QWORD); // rbx
  __int64 v13; // rax
  __int64 v14; // rax

  v8 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 96))(a1);
  sub_1F45FE0(v8, (_BYTE *)(v8 + 225), a3);
  v9 = sub_1F45FB0((_QWORD *)v8);
  v10 = 1;
  if ( v9 )
    v10 = *(_OWORD *)(v8 + 184) == 0;
  *a4 = v10;
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, v8, 1);
  if ( !a6 )
  {
    v14 = sub_22077B0(0x710u);
    a6 = v14;
    if ( v14 )
      sub_1E2CD10(v14, a1);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, a6, 1);
  if ( (unsigned __int8)sub_1F49630((__int64 *)v8) )
    return 0;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 248LL))(v8);
  *(_BYTE *)(v8 + 224) = 1;
  if ( !*a4 )
  {
    v12 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
    v13 = sub_39B99F0(a5);
    v12(a2, v13, 0);
  }
  return a6 + 168;
}
