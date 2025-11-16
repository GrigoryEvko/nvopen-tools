// Function: sub_396F480
// Address: 0x396f480
//
void __fastcall sub_396F480(__int64 a1, unsigned int a2, __int64 a3)
{
  int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // rsi
  bool v9; // cc
  __int64 v10; // rax

  v3 = a2;
  if ( a3 )
  {
    v5 = sub_1632FA0(*(_QWORD *)(a3 + 40));
    v3 = sub_396B790(a3, v5, a2);
  }
  if ( v3 )
  {
    v6 = sub_396E9A0(a1);
    v7 = *(__int64 **)(a1 + 256);
    v8 = (unsigned int)(1 << v3);
    v9 = (unsigned __int8)(*(_BYTE *)(v6 + 148) - 1) <= 1u;
    v10 = *v7;
    if ( v9 )
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v10 + 520))(v7, v8, 0);
    else
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64, _QWORD))(v10 + 512))(v7, v8, 0, 1, 0);
  }
}
