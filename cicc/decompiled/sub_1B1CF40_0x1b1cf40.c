// Function: sub_1B1CF40
// Address: 0x1b1cf40
//
__int64 __fastcall sub_1B1CF40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r15
  _DWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = 0;
  v14[0] = sub_1560340((_QWORD *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL) + 112LL), -1, "no-nans-fp-math", 0xFu);
  v9 = (_DWORD *)sub_155D8B0(v14);
  if ( v10 == 4 )
    v6 = *v9 == 1702195828;
  if ( (unsigned __int8)sub_1B1CF10(a1, 1u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 2u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 3u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 4u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 5u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 6u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 8u, a2, v6, a3, a4, a5, a6)
    || (unsigned __int8)sub_1B1CF10(a1, 7u, a2, v6, a3, a4, a5, a6) )
  {
    return 1;
  }
  else
  {
    return sub_1B1CF10(a1, 9u, a2, v6, a3, a4, a5, a6);
  }
}
