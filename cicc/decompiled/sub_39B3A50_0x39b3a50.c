// Function: sub_39B3A50
// Address: 0x39b3a50
//
__int64 __fastcall sub_39B3A50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, char a6, __int64 a7)
{
  __int64 v10; // rax
  void (__fastcall *v11)(__int64, __int64, __int64); // rbx
  __int64 v12; // rax
  char v14; // [rsp+Fh] [rbp-31h] BYREF

  v14 = 1;
  v10 = sub_39B13D0(a1, a2, a6, (bool *)&v14, a3, a7);
  if ( !v10 || v14 && (unsigned __int8)sub_39B3200((__int64)a1, a2, a3, a4, a5, v10) )
    return 1;
  v11 = *(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL);
  v12 = sub_1E2D0B0();
  v11(a2, v12, 1);
  return 0;
}
