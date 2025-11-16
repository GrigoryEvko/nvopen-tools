// Function: sub_1E6A210
// Address: 0x1e6a210
//
__int64 __fastcall sub_1E6A210(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r13
  __int64 (*v3)(void); // rax
  __int64 result; // rax
  __int64 (*v5)(); // rcx

  v2 = 0;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v3 != sub_1D00B10 )
    v2 = v3();
  result = sub_1E69FD0(a1, a2);
  if ( !(_BYTE)result )
  {
    v5 = *(__int64 (**)())(*(_QWORD *)v2 + 80LL);
    if ( v5 != sub_1E1C7F0 )
      return ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v5)(v2, a2, *a1);
  }
  return result;
}
