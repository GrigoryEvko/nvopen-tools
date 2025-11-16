// Function: sub_3157DA0
// Address: 0x3157da0
//
unsigned __int64 __fastcall sub_3157DA0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v7; // r13
  unsigned __int64 v8; // rax

  v5 = a3 & 0x70;
  if ( !(_DWORD)v5 )
    return a2;
  if ( (_DWORD)v5 != 16 )
    sub_C64ED0("We do not support this DWARF encoding yet!", 1u);
  v7 = sub_E6C430(*(_QWORD *)(a1 + 920), a2, v5, a4, a5);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a4 + 208LL))(a4, v7, 0);
  v8 = sub_E808D0(v7, 0, *(_QWORD **)(a1 + 920), 0);
  return sub_E81A00(18, a2, v8, *(_QWORD **)(a1 + 920), 0);
}
