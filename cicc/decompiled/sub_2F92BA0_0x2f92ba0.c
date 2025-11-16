// Function: sub_2F92BA0
// Address: 0x2f92ba0
//
__int64 __fastcall sub_2F92BA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 result; // rax

  if ( a2 != a1 + 328 )
  {
    v8 = (unsigned int)sub_2F90B20(a1 + 2928, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL, a2, a4, a5, a6);
    result = 0;
    if ( (_BYTE)v8 )
      return result;
    sub_2F8FA50(a1 + 2928, a2, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL, v7, v8, v9);
  }
  if ( ((*(_BYTE *)a3 ^ 6) & 6) != 0 )
    sub_2F8F1B0(a2, a3, 1u, a4, a5, a6);
  else
    sub_2F8F1B0(a2, a3, *(_DWORD *)(a3 + 8) != 3, a4, a5, a6);
  return 1;
}
