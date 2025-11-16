// Function: sub_E76730
// Address: 0xe76730
//
__int64 __fastcall sub_E76730(__int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4)
{
  char v4; // al
  unsigned __int8 v5; // r14
  __int64 v6; // rax
  int v7; // r13d
  __int64 v8; // r15
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax

  v4 = *(_BYTE *)(a2[1] + 1906LL);
  if ( v4 )
  {
    if ( v4 != 1 )
      BUG();
    v5 = 8;
  }
  else
  {
    v5 = 4;
  }
  v6 = sub_E766E0(a1, a3, a4);
  v7 = v6;
  if ( !*(_BYTE *)(a1 + 160) )
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 536LL))(a2, v6, v5);
  v8 = a2[1];
  if ( *(_BYTE *)(*(_QWORD *)(v8 + 152) + 259LL) )
    return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 368LL))(a2, *(_QWORD *)(a1 + 104), v6);
  v10 = sub_E808D0(*(_QWORD *)(a1 + 104), 0, a2[1], 0);
  v11 = sub_E81A90(v7, v8, 0, 0);
  v12 = sub_E81A00(0, v10, v11, v8, 0);
  return sub_E9A5B0(a2, v12, v5, 0);
}
