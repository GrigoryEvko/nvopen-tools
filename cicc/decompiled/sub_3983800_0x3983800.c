// Function: sub_3983800
// Address: 0x3983800
//
void __fastcall sub_3983800(__int64 a1, __int64 a2, __int64 (*a3)(void))
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  _DWORD *v7; // rax
  unsigned int v8; // r15d
  _DWORD *v9; // r14
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 v12; // r12
  void (__fastcall *v13)(__int64, __int64, _QWORD); // rbx
  __int64 v14; // rax

  v4 = *(_QWORD *)(a1 + 8);
  if ( !*(_BYTE *)(a1 + 25) )
  {
    if ( *(_BYTE *)(v4 + 536) )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v4 + 256) + 712LL))(*(_QWORD *)(v4 + 256), 0, 1);
      v4 = *(_QWORD *)(a1 + 8);
    }
    *(_BYTE *)(a1 + 25) = 1;
  }
  sub_38E0040(*(_QWORD *)(v4 + 256), 0);
  if ( *(_BYTE *)(a1 + 26) )
  {
    v5 = sub_15E38F0(**(_QWORD **)(a2 + 56));
    v6 = sub_1649C60(v5);
    if ( *(_BYTE *)(v6 + 16) )
      v6 = 0;
    if ( *(_BYTE *)(a1 + 27) )
      sub_1E2D790(*(_QWORD **)(a1 + 16), v6);
    v7 = (_DWORD *)sub_396DD80(*(_QWORD *)(a1 + 8));
    v8 = v7[3];
    v9 = v7;
    v10 = (*(__int64 (__fastcall **)(_DWORD *, __int64, _QWORD, _QWORD))(*(_QWORD *)v7 + 88LL))(
            v7,
            v6,
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL),
            *(_QWORD *)(a1 + 16));
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 752LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      v10,
      v8);
    if ( *(_BYTE *)(a1 + 28) )
    {
      v11 = v9[4];
      v12 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
      v13 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v12 + 760LL);
      v14 = a3();
      v13(v12, v14, v11);
    }
  }
}
