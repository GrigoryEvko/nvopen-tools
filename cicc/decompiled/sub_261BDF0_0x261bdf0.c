// Function: sub_261BDF0
// Address: 0x261bdf0
//
void __fastcall __noreturn sub_261BDF0(__int64 a1, __int64 *a2, __int64 a3)
{
  int v4; // eax
  int v5; // r13d
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // [rsp+8h] [rbp-58h] BYREF
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+30h] [rbp-30h]

  if ( *(_QWORD *)(a1 + 48) )
  {
    v4 = (*(__int64 (__fastcall **)(__int64))(a1 + 56))(a1 + 32);
    v13 = a1;
    v5 = v4;
    LOWORD(v14) = 260;
    v6 = (__int64 *)sub_CB72A0();
    v7 = *a2;
    *a2 = 0;
    v12 = v7 | 1;
    sub_C63F70((unsigned __int64 *)&v12, v6, v8, v9, v10, v11, v13);
    if ( (v12 & 1) == 0 && (v12 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      exit(v5);
    sub_C63C30(&v12, (__int64)v6);
  }
  sub_4263D6(a1, a2, a3);
}
