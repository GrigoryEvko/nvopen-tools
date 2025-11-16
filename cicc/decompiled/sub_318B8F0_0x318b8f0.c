// Function: sub_318B8F0
// Address: 0x318b8f0
//
_QWORD *__fastcall sub_318B8F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned int **v12; // rax
  _BYTE *v13; // rax

  v12 = (unsigned int **)sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v13 = (_BYTE *)sub_B36550(v12, *(_QWORD *)(a1 + 16), *(_QWORD *)(a2 + 16), *(_QWORD *)(a3 + 16), a5, 0);
  if ( *v13 == 86 )
    return sub_3189990(a4, (__int64)v13);
  else
    return (_QWORD *)sub_31892C0(a4, (__int64)v13);
}
