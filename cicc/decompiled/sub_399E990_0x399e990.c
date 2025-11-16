// Function: sub_399E990
// Address: 0x399e990
//
__int64 __fastcall sub_399E990(__int64 a1, int a2, int a3)
{
  void (__fastcall *v4)(__int64, __int64, _QWORD); // rax
  void (__fastcall **v5)(__int64, __int64, _QWORD); // rax
  __int64 v6; // rsi
  void (*v8)(); // rdx

  v4 = **(void (__fastcall ***)(__int64, __int64, _QWORD))a1;
  if ( *(_BYTE *)(a1 + 80) )
  {
    v4(a1, 146, 0);
    v5 = *(void (__fastcall ***)(__int64, __int64, _QWORD))a1;
    v8 = *(void (**)())(*(_QWORD *)a1 + 8LL);
    if ( v8 != nullsub_1985 )
    {
      ((void (__fastcall *)(__int64, _QWORD))v8)(a1, (unsigned int)a2);
      v5 = *(void (__fastcall ***)(__int64, __int64, _QWORD))a1;
    }
    v6 = a3;
  }
  else
  {
    if ( a2 <= 31 )
    {
      v4(a1, (unsigned __int8)(a2 + 112), 0);
    }
    else
    {
      v4(a1, 146, 0);
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, a2);
    }
    v5 = *(void (__fastcall ***)(__int64, __int64, _QWORD))a1;
    v6 = a3;
  }
  return ((__int64 (__fastcall *)(__int64, __int64))v5[2])(a1, v6);
}
