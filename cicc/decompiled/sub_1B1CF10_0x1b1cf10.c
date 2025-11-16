// Function: sub_1B1CF10
// Address: 0x1b1cf10
//
__int64 __fastcall sub_1B1CF10(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 2 && *(_QWORD *)(a1 + 40) == **(_QWORD **)(a3 + 32) )
    return sub_1B1B260(a1, a2, a3, a4, a5, a6, a7, a8);
  else
    return 0;
}
