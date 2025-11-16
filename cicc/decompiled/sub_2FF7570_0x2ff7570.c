// Function: sub_2FF7570
// Address: 0x2ff7570
//
__int64 __fastcall sub_2FF7570(__int64 a1, _QWORD *a2, char a3, __int64 *a4)
{
  void (__fastcall *v5)(__int64, __int64 *, __int64); // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-A8h]
  _QWORD v14[20]; // [rsp+10h] [rbp-A0h] BYREF

  v12 = v14;
  v13 = 0x1000000000LL;
  if ( (a3 & 1) != 0 )
  {
    v14[0] = 6;
    LODWORD(v13) = 1;
  }
  v5 = *(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v5 == sub_2FF5450 )
    sub_AF6280((__int64)&v12, *a4);
  else
    v5(a1, a4, (__int64)&v12);
  if ( (a3 & 2) != 0 )
  {
    v10 = (unsigned int)v13;
    v11 = (unsigned int)v13 + 1LL;
    if ( v11 > HIDWORD(v13) )
    {
      sub_C8D5F0((__int64)&v12, v14, v11, 8u, v6, v7);
      v10 = (unsigned int)v13;
    }
    v12[v10] = 6;
    LODWORD(v13) = v13 + 1;
  }
  v8 = sub_B0D8A0(a2, (__int64)&v12, (a3 & 4) != 0, (a3 & 8) != 0);
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  return v8;
}
