// Function: sub_2165CB0
// Address: 0x2165cb0
//
__int64 __fastcall sub_2165CB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, _QWORD); // rbx
  __int64 v5; // rsi

  v4 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a3 + 16LL);
  v5 = sub_216F6C0(*(unsigned int *)(*(_QWORD *)a1 + 1212LL));
  return v4(a3, v5, 0);
}
