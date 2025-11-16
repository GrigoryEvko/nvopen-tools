// Function: sub_DFB300
// Address: 0xdfb300
//
__int64 __fastcall sub_DFB300(__int64 *a1, __int64 a2, char a3)
{
  __int64 v3; // rdi
  __int64 (__fastcall *v4)(__int64, __int64, char); // rax
  __int64 v6; // [rsp-8h] [rbp-8h]

  v3 = *a1;
  v4 = *(__int64 (__fastcall **)(__int64, __int64, char))(*(_QWORD *)v3 + 1056LL);
  if ( v4 != sub_DF6050 )
    return v4(v3, a2, a3);
  *((_DWORD *)&v6 - 2) = 0;
  *((_BYTE *)&v6 - 4) = a3;
  return *(&v6 - 1);
}
