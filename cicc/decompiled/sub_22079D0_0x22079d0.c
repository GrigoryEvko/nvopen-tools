// Function: sub_22079D0
// Address: 0x22079d0
//
__int64 __fastcall sub_22079D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  const char *v8; // rdi
  const char *v9; // rsi

  if ( a5 == a3
    && ((v8 = *(const char **)(a1 + 8), v9 = *(const char **)(a4 + 8), v8 == v9) || *v8 != 42 && !strcmp(v8, v9)) )
  {
    return 6;
  }
  else
  {
    return (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 64LL))(*(_QWORD *)(a1 + 16), a2);
  }
}
