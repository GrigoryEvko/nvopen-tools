// Function: sub_74A2C0
// Address: 0x74a2c0
//
void __fastcall sub_74A2C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // r12
  void (__fastcall *v5)(_QWORD, __int64, _QWORD); // rax
  __int64 v6; // rdi
  void (__fastcall *v7)(__int64, __int64, _QWORD); // rax

  for ( i = a1; a2 != i; i = *(_QWORD *)(i + 160) )
  {
    while ( *(_BYTE *)(i + 184) != 8 )
    {
LABEL_3:
      i = *(_QWORD *)(i + 160);
      if ( a2 == i )
        goto LABEL_7;
    }
    v5 = *(void (__fastcall **)(_QWORD, __int64, _QWORD))(a3 + 88);
    if ( !v5 )
    {
      sub_74A260(i, (void (__fastcall **)(char *))a3);
      goto LABEL_3;
    }
    v5(*(_QWORD *)(i + 104), 18, 0);
  }
LABEL_7:
  if ( *(_BYTE *)(i + 140) == 7 )
  {
    v6 = *(_QWORD *)(i + 104);
    if ( v6 )
    {
      v7 = *(void (__fastcall **)(__int64, __int64, _QWORD))(a3 + 88);
      if ( v7 )
        v7(v6, 18, 0);
      else
        sub_74A260(i, (void (__fastcall **)(char *))a3);
    }
  }
}
