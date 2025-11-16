// Function: sub_359A2D0
// Address: 0x359a2d0
//
void __fastcall sub_359A2D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  __int64 *v10; // rdi
  __int64 v11; // rax
  __int64 (*v12)(); // rax
  __int64 v13; // rdi
  void (__fastcall *v14)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD); // rax
  void (__fastcall *v15)(__int64 *, __int64, __int64, __int64, _BYTE *, _QWORD); // rax
  __int64 v16; // [rsp+18h] [rbp-E8h]
  _BYTE *v17; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v18; // [rsp+28h] [rbp-D8h]
  _BYTE v19[208]; // [rsp+30h] [rbp-D0h] BYREF

  v9 = *(_QWORD *)(a1 + 120);
  v17 = v19;
  v18 = 0x400000000LL;
  (*(void (__fastcall **)(__int64, _QWORD, __int64, _BYTE **, __int64))(*(_QWORD *)v9 + 40LL))(v9, a3, a2, &v17, a4);
  v10 = *(__int64 **)(a1 + 32);
  v11 = *v10;
  if ( (_BYTE)qword_503FCA8 )
  {
    v12 = *(__int64 (**)())(v11 + 880);
    if ( v12 == sub_2DB1B20 || ((unsigned __int8 (__fastcall *)(__int64 *, _BYTE **))v12)(v10, &v17) )
      BUG();
    v13 = *(_QWORD *)(a1 + 32);
    v14 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD))(*(_QWORD *)v13 + 368LL);
    v16 = 0;
    v14(v13, a2, a6, a5, v17, (unsigned int)v18);
  }
  else
  {
    v15 = *(void (__fastcall **)(__int64 *, __int64, __int64, __int64, _BYTE *, _QWORD))(v11 + 368);
    v16 = 0;
    v15(v10, a2, a5, a6, v17, (unsigned int)v18);
  }
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
}
