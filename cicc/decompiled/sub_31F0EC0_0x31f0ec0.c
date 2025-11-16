// Function: sub_31F0EC0
// Address: 0x31f0ec0
//
__int64 __fastcall sub_31F0EC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, __int64, __int64, _QWORD); // rbx
  __int64 v5; // rcx

  v4 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 432LL);
  v5 = (unsigned int)sub_31DF6B0(a1);
  return v4(a1, a2, a3, v5, 0);
}
