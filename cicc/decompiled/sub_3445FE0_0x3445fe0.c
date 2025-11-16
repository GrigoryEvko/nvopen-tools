// Function: sub_3445FE0
// Address: 0x3445fe0
//
unsigned __int8 *__fastcall sub_3445FE0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        unsigned int a9)
{
  __int128 v11; // rax
  __int64 v12; // r9
  __int128 v14; // [rsp-50h] [rbp-80h]
  __int128 v15; // [rsp+0h] [rbp-30h]

  *(_QWORD *)&v15 = a5;
  *((_QWORD *)&v15 + 1) = a6;
  *(_QWORD *)&v11 = sub_33ED040(a1, a9);
  *((_QWORD *)&v14 + 1) = a4;
  *(_QWORD *)&v14 = a3;
  return sub_33FC1D0(
           a1,
           207,
           a2,
           *(unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * DWORD2(a7)),
           *(_QWORD *)(*(_QWORD *)(a7 + 48) + 16LL * DWORD2(a7) + 8),
           v12,
           v14,
           v15,
           a7,
           a8,
           v11);
}
