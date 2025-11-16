// Function: sub_13671D0
// Address: 0x13671d0
//
__int64 __fastcall sub_13671D0(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __int64 a7,
        __int64 a8)
{
  char v11; // r13
  __int64 result; // rax

  if ( *(_BYTE *)(a5 + 16) == 79 && *(_QWORD *)(a2 - 72) == *(_QWORD *)(a5 - 72) )
  {
    v11 = sub_1362890(
            a1,
            *(_QWORD *)(a2 - 48),
            a3,
            *(_QWORD *)(a5 - 48),
            a6,
            0,
            *(_OWORD *)a4,
            *(_QWORD *)(a4 + 16),
            *(_OWORD *)a7,
            *(_QWORD *)(a7 + 16),
            0);
    if ( v11 == 1 )
      return 1;
    result = sub_1362890(
               a1,
               *(_QWORD *)(a2 - 24),
               a3,
               *(_QWORD *)(a5 - 24),
               a6,
               0,
               *(_OWORD *)a4,
               *(_QWORD *)(a4 + 16),
               *(_OWORD *)a7,
               *(_QWORD *)(a7 + 16),
               0);
  }
  else
  {
    v11 = sub_1362890(
            a1,
            a5,
            a6,
            *(_QWORD *)(a2 - 48),
            a3,
            a8,
            *(_OWORD *)a7,
            *(_QWORD *)(a7 + 16),
            *(_OWORD *)a4,
            *(_QWORD *)(a4 + 16),
            0);
    if ( v11 == 1 )
      return 1;
    result = sub_1362890(
               a1,
               a5,
               a6,
               *(_QWORD *)(a2 - 24),
               a3,
               a8,
               *(_OWORD *)a7,
               *(_QWORD *)(a7 + 16),
               *(_OWORD *)a4,
               *(_QWORD *)(a4 + 16),
               0);
  }
  if ( v11 == (_BYTE)result )
    return result;
  if ( (_BYTE)result == 2 && v11 == 3 || v11 == 2 && (_BYTE)result == 3 )
    return 2;
  return 1;
}
