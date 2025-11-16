// Function: sub_2FEBB30
// Address: 0x2febb30
//
__int64 __fastcall sub_2FEBB30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _DWORD *a7)
{
  __int64 (__fastcall *v9)(__int64, __int64, __int64, __int64, __int64, unsigned int, unsigned __int8, unsigned __int16, _DWORD *); // rbx
  __int64 v10; // r9
  int v14; // [rsp+18h] [rbp-38h]
  unsigned __int8 v15; // [rsp+1Fh] [rbp-31h]

  v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, unsigned int, unsigned __int8, unsigned __int16, _DWORD *))(*(_QWORD *)a1 + 824LL);
  v14 = *(unsigned __int16 *)(a6 + 32);
  v15 = sub_2EAC4F0(a6);
  v10 = (unsigned int)sub_2EAC1E0(a6);
  if ( v9 == sub_2FEBA90 )
    return sub_2FEB980(a1, a2, a3, a4, a5, v10, v15, v14, a7);
  else
    return v9(a1, a2, a3, a4, a5, v10, v15, v14, a7);
}
