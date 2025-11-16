// Function: sub_3221AB0
// Address: 0x3221ab0
//
__int64 __fastcall sub_3221AB0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        _BYTE *a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int16 v11; // ax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v13 = *(_QWORD *)(a1 + 3232);
  v11 = sub_3220AA0(a1);
  return sub_321C5E0(
           *(_QWORD ***)(a1 + 8),
           a2,
           a3,
           a4,
           a5,
           *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL) + 1912LL),
           v11,
           v13,
           a7,
           a8);
}
