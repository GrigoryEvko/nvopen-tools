// Function: sub_22201B0
// Address: 0x22201b0
//
__int64 __fastcall sub_22201B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7)
{
  const wchar_t *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v17[2]; // [rsp+10h] [rbp-60h] BYREF
  char v18; // [rsp+20h] [rbp-50h] BYREF
  void (__fastcall *v19)(unsigned __int64 *); // [rsp+30h] [rbp-40h]

  v17[0] = (__int64)&v18;
  v10 = *(const wchar_t **)a7;
  v11 = *(_QWORD *)(a7 + 8);
  v19 = 0;
  sub_2220100(v17, v10, (__int64)&v10[v11]);
  v12 = *(_QWORD *)(a1 + 16);
  v19 = sub_221F8F0;
  v13 = sub_2214740(v12, a2, a3, a4, a5, a6, 0, v17);
  if ( v19 )
    v19((unsigned __int64 *)v17);
  return v13;
}
