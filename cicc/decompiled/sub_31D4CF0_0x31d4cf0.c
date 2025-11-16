// Function: sub_31D4CF0
// Address: 0x31d4cf0
//
void __fastcall sub_31D4CF0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, char a5)
{
  unsigned __int8 *v7; // r13
  _QWORD *v8; // r15
  unsigned __int64 v9; // rax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 259LL) && a5 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 368LL))(*(_QWORD *)(a1 + 224));
    if ( a4 > 4 )
      sub_E99300(*(_QWORD ***)(a1 + 224), a4 - 4);
  }
  else
  {
    v7 = (unsigned __int8 *)sub_E808D0(a2, 0, *(_QWORD **)(a1 + 216), 0);
    if ( a3 )
    {
      v8 = *(_QWORD **)(a1 + 216);
      v9 = sub_E81A90(a3, v8, 0, 0);
      v7 = (unsigned __int8 *)sub_E81A00(0, (__int64)v7, v9, v8, 0);
    }
    sub_E9A5B0(*(_QWORD *)(a1 + 224), v7);
  }
}
