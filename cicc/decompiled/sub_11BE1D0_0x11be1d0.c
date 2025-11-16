// Function: sub_11BE1D0
// Address: 0x11be1d0
//
__int64 __fastcall sub_11BE1D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned __int64 a8)
{
  __int64 result; // rax

  result = sub_98CF40(a2, *(_QWORD *)(*(_QWORD *)a1 + 424LL), *(_QWORD *)(*(_QWORD *)a1 + 440LL), 0);
  if ( !(_BYTE)result )
    return 0;
  if ( a8 >= *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) )
  {
    **(_BYTE **)(a1 + 16) = 1;
    return result;
  }
  result = sub_98CF40(*(_QWORD *)(*(_QWORD *)a1 + 424LL), a2, *(_QWORD *)(*(_QWORD *)a1 + 440LL), 0);
  if ( !(_BYTE)result )
    return 0;
  **(_BYTE **)(a1 + 16) = 1;
  **(_QWORD **)(a1 + 24) = a2
                         + 32
                         * ((unsigned int)(*(_DWORD *)(a3 + 8) + 1) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  return result;
}
