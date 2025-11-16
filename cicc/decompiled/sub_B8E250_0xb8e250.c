// Function: sub_B8E250
// Address: 0xb8e250
//
_QWORD *__fastcall sub_B8E250(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9)
{
  __int64 v9; // r9
  __int64 v11; // r8
  __int64 i; // rcx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  _QWORD *v16; // rdx
  _QWORD *v17; // rax

  v9 = a2;
  v11 = (a3 - 1) / 2;
  if ( a2 < v11 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * (i + 1);
      v13 = (_QWORD *)(a1 + 48 * (i + 1));
      if ( v13[2] < *(v13 - 1) )
      {
        --a2;
        v13 = (_QWORD *)(a1 + 24 * a2);
      }
      v14 = (_QWORD *)(a1 + 24 * i);
      *v14 = *v13;
      a4 = v13[1];
      v14[1] = a4;
      v14[2] = v13[2];
      if ( a2 >= v11 )
        break;
    }
  }
  if ( (a3 & 1) != 0 || (a3 - 2) / 2 != a2 )
    return sub_B8E1B0(a1, a2, v9, a4, v11, v9, a7, a8, a9);
  v16 = (_QWORD *)(a1 + 48 * (a2 + 1) - 24);
  v17 = (_QWORD *)(a1 + 24 * a2);
  *v17 = *v16;
  v17[1] = v16[1];
  v17[2] = v16[2];
  return sub_B8E1B0(a1, 2 * (a2 + 1) - 1, v9, 2 * (a2 + 1), v11, v9, a7, a8, a9);
}
