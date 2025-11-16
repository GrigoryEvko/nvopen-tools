// Function: sub_2CEABE0
// Address: 0x2ceabe0
//
unsigned __int64 __fastcall sub_2CEABE0(
        _QWORD *a1,
        __int64 a2,
        unsigned __int8 *a3,
        __int64 a4,
        _QWORD *a5,
        int a6,
        unsigned __int8 a7)
{
  __int64 v7; // rax
  int v8; // r9d

  v7 = (unsigned int)(a6 - 1);
  v8 = 0;
  if ( (unsigned int)v7 <= 0x1F )
    v8 = byte_444AF40[v7];
  return sub_2CE9A90(a1, a2, a3, a4, a5, v8, a7);
}
