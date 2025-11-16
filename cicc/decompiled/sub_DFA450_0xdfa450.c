// Function: sub_DFA450
// Address: 0xdfa450
//
__int64 __fastcall sub_DFA450(__int64 **a1, __int64 a2, int a3)
{
  char v3; // r12
  __int64 *v4; // rdi
  __int64 (__fastcall *v5)(__int64, __int64, char); // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rdx
  _QWORD v11[3]; // [rsp+0h] [rbp-20h] BYREF

  v3 = a3;
  v4 = *a1;
  v5 = *(__int64 (__fastcall **)(__int64, __int64, char))(*v4 + 544);
  if ( v5 != sub_DF7330 )
    return v5((__int64)v4, a2, a3);
  v6 = sub_9208B0(v4[1], a2);
  v11[1] = v7;
  v11[0] = (unsigned __int64)(v6 + 7) >> 3;
  v8 = sub_CA1930(v11);
  v9 = 1LL << v3;
  LOBYTE(v9) = v8 != 0 && v8 <= (unsigned __int64)(1LL << v3);
  if ( (_BYTE)v9 )
  {
    LODWORD(v9) = v8 - 1;
    LOBYTE(v9) = (v8 & (v8 - 1)) == 0;
  }
  return (unsigned int)v9;
}
