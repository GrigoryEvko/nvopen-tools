// Function: sub_2526260
// Address: 0x2526260
//
__int64 __fastcall sub_2526260(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, unsigned __int64, __int64),
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        _BYTE *a6,
        int *a7,
        __int64 a8,
        char a9,
        unsigned __int8 a10)
{
  __int64 v14; // r9
  __int64 v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v18[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( !a4 || sub_B2FC80(a4) )
    return 0;
  sub_250D230(v18, a4, 4, 0);
  if ( a5 && a10 )
    v14 = sub_251BBC0(a1, v18[0], v18[1], a5, 2, 0, 1);
  else
    v14 = 0;
  v16 = (_QWORD *)v14;
  v15 = sub_251B1C0(*(_QWORD *)(a1 + 208), a4);
  return sub_2522A30(a1, v15, a2, a3, a5, v16, a7, a8, a6, a9, a10);
}
