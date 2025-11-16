// Function: sub_34CD4A0
// Address: 0x34cd4a0
//
__int64 __fastcall sub_34CD4A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, char a6, __int64 a7)
{
  __int64 v9; // rbx
  __int64 v10; // r9
  __int64 v11; // rbx
  void (__fastcall *v12)(__int64, __int64, __int64); // rbx
  __int64 v13; // rax
  void (__fastcall *v15)(__int64, __int64, _QWORD); // rbx
  __int64 v16; // rax
  _QWORD *v17; // rax
  char v18; // [rsp+4h] [rbp-3Ch]

  v9 = a7;
  if ( !a7 )
  {
    v18 = a6;
    v17 = (_QWORD *)sub_22077B0(0xAB0u);
    a6 = v18;
    v9 = (__int64)v17;
    if ( v17 )
    {
      sub_2EAA600(v17, a1);
      a6 = v18;
    }
  }
  if ( sub_34CD400((__int64)a1, a2, a6, v9) )
  {
    if ( !sub_2FF0720() )
    {
      if ( a5 != 2 )
      {
        v15 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
        v16 = sub_2F25A00(a3);
        v15(a2, v16, 0);
      }
      goto LABEL_7;
    }
    v10 = *(_QWORD *)(v9 + 2656);
    v11 = v9 + 184;
    if ( !v10 )
      v10 = v11;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __int64, __int64, _QWORD, __int64))(*a1 + 208))(
            a1,
            a2,
            a3,
            a4,
            a5,
            v10) )
    {
LABEL_7:
      v12 = *(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL);
      v13 = sub_2EAA4A0();
      v12(a2, v13, 1);
      return 0;
    }
  }
  return 1;
}
