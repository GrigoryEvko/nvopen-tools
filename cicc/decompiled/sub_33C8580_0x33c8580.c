// Function: sub_33C8580
// Address: 0x33c8580
//
void __fastcall sub_33C8580(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  const char *v5; // [rsp-38h] [rbp-38h] BYREF
  int v6; // [rsp-28h] [rbp-28h]
  __int16 v7; // [rsp-18h] [rbp-18h]
  __int64 v8; // [rsp-8h] [rbp-8h]

  if ( (_DWORD)a2 )
  {
    v8 = v2;
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(__int64 (**)())(*(_QWORD *)v3 + 80LL);
    if ( v4 == sub_23CE2F0 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v4)(v3, a2, 0) )
    {
      v6 = a2;
      v5 = "cannot lower memory intrinsic in address space ";
      v7 = 2307;
      sub_C64D30((__int64)&v5, 1u);
    }
  }
}
