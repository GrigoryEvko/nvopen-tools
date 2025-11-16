// Function: sub_27F0080
// Address: 0x27f0080
//
__int64 __fastcall sub_27F0080(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8,
        char a9,
        unsigned __int8 a10)
{
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // r12d
  __int64 v16; // rsi
  unsigned __int8 *v18; // [rsp+8h] [rbp-38h] BYREF

  if ( a9 && sub_991A70(a1, a7, a8, a2, a3, 1u, a10) )
    return 1;
  v15 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 *, __int64, __int64))(*(_QWORD *)a5 + 24LL))(a5, a1, a2, a4);
  if ( (_BYTE)v15 )
  {
    return 1;
  }
  else if ( *a1 == 61 )
  {
    v16 = *((_QWORD *)a1 - 4);
    v18 = a1;
    if ( (unsigned __int8)sub_D48480(a4, v16, v13, v14) )
      sub_27EFD70(a6, (__int64 *)&v18);
  }
  return v15;
}
