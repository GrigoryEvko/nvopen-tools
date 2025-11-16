// Function: sub_39826E0
// Address: 0x39826e0
//
void __fastcall sub_39826E0(__int64 *a1, __int64 a2, __int16 a3)
{
  __int64 v4; // r15
  __int64 (***v5)(); // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // r14
  void (__fastcall *v8)(__int64, __int64, _QWORD); // rbx
  unsigned int v9; // eax
  __int64 v10; // r14
  void (__fastcall *v11)(__int64, _QWORD, _QWORD); // r15
  unsigned int v12; // eax
  __int64 v13; // r14
  unsigned int v14; // eax

  if ( (unsigned __int16)a3 > 0x14u )
  {
    sub_397C0C0(a2, *(unsigned int *)(*a1 + 16), 0);
  }
  else if ( (unsigned __int16)a3 > 0x10u )
  {
    v10 = *(_QWORD *)(a2 + 256);
    v11 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v10 + 424LL);
    v12 = sub_3982640((__int64)a1, a2, a3);
    v11(v10, *(unsigned int *)(*a1 + 16), v12);
  }
  else
  {
    v4 = (unsigned int)sub_3981F00(*a1);
    v5 = (__int64 (***)())sub_3981ED0(*a1);
    v6 = **v5;
    if ( v6 == sub_3981A10 || (v13 = ((__int64 (__fastcall *)(__int64 (***)()))v6)(v5)) == 0 )
    {
      v7 = *(_QWORD *)(a2 + 256);
      v8 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 424LL);
      v9 = sub_3982640((__int64)a1, a2, 16);
      v8(v7, v4, v9);
    }
    else
    {
      v14 = sub_3982640((__int64)a1, a2, 16);
      sub_396F390(a2, v13, v4, v14, 1);
    }
  }
}
