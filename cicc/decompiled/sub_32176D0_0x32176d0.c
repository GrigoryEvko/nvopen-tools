// Function: sub_32176D0
// Address: 0x32176d0
//
void __fastcall sub_32176D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // edx
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // r12
  _DWORD *v10; // rax
  unsigned int v11; // r15d
  _DWORD *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // r13d
  __int64 v16; // r12
  void (__fastcall *v17)(__int64, __int64, _QWORD); // rbx
  __int64 v18; // rax

  if ( *(_BYTE *)(a1 + 27) )
  {
    v6 = *(_QWORD *)(a1 + 8);
    if ( !*(_BYTE *)(a1 + 28) )
    {
      v7 = *(_DWORD *)(v6 + 776);
      if ( v7 == 2 || (a4 = *(_QWORD *)(v6 + 200), (*(_BYTE *)(a4 + 904) & 0x10) != 0) )
      {
        (*(void (__fastcall **)(_QWORD, bool, __int64))(**(_QWORD **)(v6 + 224) + 856LL))(
          *(_QWORD *)(v6 + 224),
          v7 == 1,
          1);
        v6 = *(_QWORD *)(a1 + 8);
      }
      *(_BYTE *)(a1 + 28) = 1;
    }
    sub_E9C600(*(__int64 **)(v6 + 224), 0, 0, a4, a5, a6);
    if ( *(_BYTE *)(a1 + 24) )
    {
      v8 = (unsigned __int8 *)sub_B2E500(**(_QWORD **)(a2 + 32));
      v9 = sub_BD3990(v8, 0);
      if ( *v9 >= 4u )
        v9 = 0;
      sub_32175A0((_QWORD *)a1, (__int64)v9);
      v10 = (_DWORD *)sub_31DA6B0(*(_QWORD *)(a1 + 8));
      v11 = v10[235];
      v12 = v10;
      v13 = (*(__int64 (__fastcall **)(_DWORD *, unsigned __int8 *, _QWORD, _QWORD))(*(_QWORD *)v10 + 144LL))(
              v10,
              v9,
              *(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL),
              *(_QWORD *)(a1 + 16));
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 904LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
        v13,
        v11);
      if ( *(_BYTE *)(a1 + 26) )
      {
        v14 = *(_QWORD *)(a1 + 8);
        v15 = v12[236];
        v16 = *(_QWORD *)(v14 + 224);
        v17 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v16 + 912LL);
        v18 = sub_31E4810(v14, a2);
        v17(v16, v18, v15);
      }
    }
  }
}
