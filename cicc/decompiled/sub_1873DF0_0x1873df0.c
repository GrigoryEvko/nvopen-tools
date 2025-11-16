// Function: sub_1873DF0
// Address: 0x1873df0
//
void __fastcall __noreturn sub_1873DF0(__int64 a1, __int64 *a2, __int64 a3)
{
  int v4; // eax
  int v5; // r13d
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-48h] BYREF
  __int64 v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+20h] [rbp-30h]

  if ( *(_QWORD *)(a1 + 48) )
  {
    v4 = (*(__int64 (__fastcall **)(__int64))(a1 + 56))(a1 + 32);
    v14 = a1;
    v5 = v4;
    LOWORD(v15) = 260;
    v6 = (__int64 *)sub_16E8CB0();
    v7 = *a2;
    *a2 = 0;
    v13 = v7 | 1;
    sub_16BCD30((unsigned __int64 *)&v13, v6, v8, v9, v10, v11, v14);
    if ( (v13 & 1) == 0 && (v13 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      exit(v5);
    sub_16BCAE0(&v13, (__int64)v6, v12);
  }
  sub_4263D6(a1, a2, a3);
}
