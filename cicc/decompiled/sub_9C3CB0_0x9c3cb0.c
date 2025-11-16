// Function: sub_9C3CB0
// Address: 0x9c3cb0
//
__int64 __fastcall sub_9C3CB0(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        int a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v13; // rax

  if ( a3 == *((_DWORD *)a2 + 2) )
    return 0;
  v8 = a3;
  v9 = *a2;
  v10 = *(_QWORD *)(*a2 + 8 * v8);
  v11 = (unsigned int)v10;
  if ( *(_BYTE *)(a1 + 1832) )
    v11 = (unsigned int)(a4 - v10);
  if ( !a5 || *(_BYTE *)(a5 + 8) != 9 )
    return sub_A14C90(a1 + 744, v11, a5, a6, a7);
  v13 = sub_A12C40(a1 + 808, v11, v9, a6, a7);
  return sub_B9F6F0(*(_QWORD *)a5, v13);
}
