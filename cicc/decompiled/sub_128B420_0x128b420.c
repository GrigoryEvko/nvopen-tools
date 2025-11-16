// Function: sub_128B420
// Address: 0x128b420
//
__int64 __fastcall sub_128B420(
        __int64 a1,
        _QWORD *a2,
        unsigned __int8 a3,
        __int64 a4,
        unsigned __int8 a5,
        char a6,
        _DWORD *a7)
{
  __int64 v7; // rax
  __int64 v9[4]; // [rsp+8h] [rbp-20h] BYREF

  v9[1] = a1 + 48;
  v7 = *(_QWORD *)(a1 + 40);
  v9[0] = a1;
  v9[2] = v7;
  return sub_128A450(v9, a2, a3, a4, a5, a6, a7);
}
