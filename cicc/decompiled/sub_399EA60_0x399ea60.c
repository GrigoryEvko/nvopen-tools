// Function: sub_399EA60
// Address: 0x399ea60
//
void __fastcall sub_399EA60(__int64 a1, unsigned int a2, unsigned int a3)
{
  void (__fastcall *v4)(__int64, __int64, _QWORD); // rax

  if ( a2 )
  {
    v4 = **(void (__fastcall ***)(__int64, __int64, _QWORD))a1;
    if ( a3 | a2 & 7 )
    {
      v4(a1, 157, 0);
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, a2);
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, a3);
    }
    else
    {
      v4(a1, 147, 0);
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, a2 >> 3);
    }
    *(_QWORD *)(a1 + 56) += a2;
  }
}
