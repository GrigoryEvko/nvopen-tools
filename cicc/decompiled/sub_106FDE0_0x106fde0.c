// Function: sub_106FDE0
// Address: 0x106fde0
//
__int64 __fastcall sub_106FDE0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v9; // rdi
  _QWORD *v10; // rsi
  const char *v11; // [rsp+0h] [rbp-30h] BYREF
  char v12; // [rsp+20h] [rbp-10h]
  char v13; // [rsp+21h] [rbp-Fh]

  if ( a7 || !a8 )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, __int64, __int64))(**(_QWORD **)(a1 + 104)
                                                                                            + 32LL))(
             *(_QWORD *)(a1 + 104),
             a1,
             a2,
             a3,
             a4,
             a5);
  v9 = *a2;
  v10 = *(_QWORD **)(a4 + 16);
  v13 = 1;
  v12 = 3;
  v11 = "unsupported relocation expression";
  return sub_E66880(v9, v10, (__int64)&v11);
}
