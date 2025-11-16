// Function: sub_D138A0
// Address: 0xd138a0
//
bool __fastcall sub_D138A0(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v3; // rbp
  bool (__fastcall *v4)(__int64, unsigned __int8 *, __int64); // rax
  char v6; // [rsp-Ah] [rbp-Ah] BYREF
  char v7; // [rsp-9h] [rbp-9h] BYREF
  __int64 v8; // [rsp-8h] [rbp-8h]

  v4 = *(bool (__fastcall **)(__int64, unsigned __int8 *, __int64))(*(_QWORD *)*a1 + 40LL);
  if ( v4 != sub_D13610 )
    return v4(*a1, a2, a3);
  v8 = v3;
  return sub_BD4FF0(a2, a3, &v6, &v7) != 0;
}
