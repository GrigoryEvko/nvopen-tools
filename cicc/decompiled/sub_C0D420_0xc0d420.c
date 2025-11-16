// Function: sub_C0D420
// Address: 0xc0d420
//
__int64 __fastcall sub_C0D420(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // r12
  void (__fastcall *v9)(__int64, __int64, __int64); // rax
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = *a3;
  *a3 = 0;
  v11[0] = v7;
  v8 = sub_E551A0(a2, v11, a4, a5, a6);
  if ( v11[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
  v9 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 192);
  if ( v9 )
    v9(v8, v7, a4);
  return v8;
}
